/**
 * Instant Feedback System for Paper Writing
 *
 * 提供实时反馈，在写作过程中的关键点（subsection 完成）
 * 检测违规和质量问题，而不是等到 self_review。
 *
 * 架构：
 * - analyzeSubsectionQuality(): 对 subsection 进行轻量级检查
 * - checkProseViolations(): 检查 PROSE 规则（20+ 条）
 * - checkStructureViolations(): 检查结构问题（heading, completeness）
 * - checkCitationIssues(): 检查引用格式
 *
 * 返回：{scores, violations, suggestions}
 */

'use strict';

const fs = require('fs');
const path = require('path');

// ---------------------------------------------------------------------------
// 规则定义（轻量级，Phase 2 检查的子集）
// ---------------------------------------------------------------------------

const PROSE_RULES = {
  'PROSE.INTENSIFIERS': {
    // Aligned with policy/rules: very, extremely, highly, remarkably, substantially
    // NOTE: 'significantly' excluded — valid in statistical contexts (p < 0.05, statistically significant)
    pattern: /\b(very|extremely|highly|remarkably|substantially)\b/gi,
    message: '删除空洞强调词',
    severity: 'warn',
  },
  'PROSE.EM_DASH': {
    pattern: /—/g,  // em dash
    message: '禁止 em-dash，用 -- 或 \\textemdash{}',
    severity: 'error',
  },
  'PROSE.FILLER_PHRASES': {
    // Aligned with policy/rules: expanded list
    pattern: /\b(as mentioned|in fact|it should be noted|it is worth noting|furthermore|moreover|in addition|in order to|it is important to note|plays a crucial role)\b/gi,
    message: '删除冗余填充短语',
    severity: 'warn',
  },
  'PROSE.VAGUE_QUANTIFIERS': {
    // Aligned with policy/rules: some, many, several, a number of, a wide range of
    pattern: /\b(some|many|several|a number of|a wide range of|quite a few|a lot of)\b/gi,
    message: '避免模糊量词，使用具体数字',
    severity: 'warn',
  },
  'PROSE.SENTENCE_LENGTH': {
    // 检查句子长度（单词数）
    checker: (text) => {
      const sentences = text.match(/[^.!?]*[.!?]/g) || [];
      const violations = [];
      sentences.forEach((sent, idx) => {
        const wordCount = sent.trim().split(/\s+/).length;
        if (wordCount > 35) {
          violations.push(`Sentence ${idx + 1}: ${wordCount} words (max 35)`);
        }
      });
      return violations;
    },
    message: '单句不超过 35 词',
    severity: 'warn',
  },
  'PROSE.INFORMAL_VOCABULARY': {
    // Aligned with policy/rules: stuff, thing, things, basically, gonna, wanna, kind of, sort of
    pattern: /\b(stuff|things?|basically|gonna|wanna|kind of|sort of)\b/gi,
    message: '禁止口语化用词',
    severity: 'error',
  },
  'PROSE.ABBREVIATION_FIRST_USE': {
    // 简化版：检查是否有大写缩写（需要首次展开）
    checker: (text) => {
      const abbreviations = text.match(/\b[A-Z]{2,}\b/g) || [];
      const unique = [...new Set(abbreviations)];
      return unique.length > 0 ? [`Found ${unique.length} abbreviations - verify first use expansion`] : [];
    },
    message: '缩写首次出现时需展开',
    severity: 'warn',
  },
};

const STRUCTURE_RULES = {
  'STRUCT.SUBSECTION_COUNT': {
    checker: (text) => {
      // 检查 subsection 是否有至少 2 个段落
      const paragraphs = text.split(/\n\n+/).filter(p => p.trim().length > 50);
      if (paragraphs.length < 2) {
        return ['Subsection too short - need at least 2 paragraphs'];
      }
      return [];
    },
    message: 'Subsection 至少 2 段',
    severity: 'warn',
  },
  'STRUCT.MIN_WORD_COUNT': {
    checker: (text) => {
      const wordCount = text.split(/\s+/).length;
      if (wordCount < 100) {
        return [`Content too brief: ${wordCount} words (min 100)`];
      }
      return [];
    },
    message: 'Subsection 至少 100 词',
    severity: 'warn',
  },
};

const CITATION_RULES = {
  'CITE.BRACKET_FORMAT': {
    pattern: /\\cite\s*\{[^}]+\}/g,  // Match \cite{...}
    checker: (text) => {
      // 检查引用是否有括号格式错误
      const citations = text.match(/\\cite\s*{[^}]+}/g) || [];
      if (citations.length === 0) {
        return [];  // 没有引用不算错误
      }
      return [];
    },
    message: '引用格式检查',
    severity: 'info',
  },
};

// ---------------------------------------------------------------------------
// 分析函数
// ---------------------------------------------------------------------------

/**
 * 检查 Prose violations（返回违规列表）
 * @param {string} text - 要检查的文本
 * @param {string[]} rulesToCheck - 要检查的规则名（可选，默认全部）
 * @returns {Array<{rule: string, matches: number, examples: string[]}>}
 */
function checkProseViolations(text, rulesToCheck) {
  const violations = [];
  const rulesToApply = rulesToCheck || Object.keys(PROSE_RULES);

  for (const ruleName of rulesToApply) {
    if (!PROSE_RULES[ruleName]) continue;

    const rule = PROSE_RULES[ruleName];
    let matches = [];

    if (rule.pattern) {
      matches = text.match(rule.pattern) || [];
    } else if (rule.checker) {
      matches = rule.checker(text);
    }

    if (matches.length > 0) {
      violations.push({
        rule: ruleName,
        severity: rule.severity,
        message: rule.message,
        count: matches.length,
        examples: Array.isArray(matches) ? matches.slice(0, 3) : [],
      });
    }
  }

  return violations;
}

/**
 * 检查结构违规（返回违规列表）
 * @param {string} text - 要检查的文本
 * @returns {Array}
 */
function checkStructureViolations(text) {
  const violations = [];

  for (const ruleName of Object.keys(STRUCTURE_RULES)) {
    const rule = STRUCTURE_RULES[ruleName];
    const issues = rule.checker(text);

    if (issues.length > 0) {
      violations.push({
        rule: ruleName,
        severity: rule.severity,
        message: rule.message,
        issues,
      });
    }
  }

  return violations;
}

/**
 * 分析 subsection 质量（主入口）
 * @param {{text: string, subsectionType?: string, title?: string}} opts
 * @returns {{
 *   scores: {clarity, completeness, prose_quality},
 *   violations: {prose: [], structure: []},
 *   suggestions: string[],
 *   overall_quality: number (0-1.0)
 * }}
 */
function analyzeSubsectionQuality({ text, subsectionType, title } = {}) {
  if (!text || text.trim().length === 0) {
    return {
      scores: { clarity: 0.0, completeness: 0.0, prose_quality: 0.0 },
      violations: { prose: [], structure: [] },
      suggestions: ['Content is empty or too short'],
      overall_quality: 0.0,
    };
  }

  // 检查各类违规
  const proseViolations = checkProseViolations(text);
  const structureViolations = checkStructureViolations(text);

  // 计算评分
  const wordCount = text.split(/\s+/).length;
  const sentenceCount = (text.match(/[.!?]/g) || []).length;
  const avgSentenceLength = wordCount / Math.max(sentenceCount, 1);

  const scores = {
    clarity: Math.min(1.0, 1.0 - proseViolations.length * 0.1),  // 每个违规 -0.1
    completeness: Math.min(1.0, wordCount / 300),  // 300 词为满分
    prose_quality: Math.min(1.0, 1.0 - proseViolations.length * 0.15),
  };

  // 计算整体质量分（加权）
  const overall_quality = 0.4 * scores.clarity + 0.3 * scores.completeness + 0.3 * scores.prose_quality;

  // 生成建议
  const suggestions = [];
  if (proseViolations.length > 0) {
    suggestions.push(`Prose issues: ${proseViolations.map(v => v.rule).join(', ')}`);
  }
  if (structureViolations.length > 0) {
    suggestions.push(`Structure issues: ${structureViolations.map(v => v.rule).join(', ')}`);
  }
  if (wordCount < 100) {
    suggestions.push(`Content too brief (${wordCount} words, min 100)`);
  }
  if (avgSentenceLength > 35) {
    suggestions.push(`Sentence too long (avg ${Math.round(avgSentenceLength)} words, max 35)`);
  }

  return {
    scores,
    violations: {
      prose: proseViolations,
      structure: structureViolations,
    },
    suggestions,
    overall_quality: Math.round(overall_quality * 100) / 100,  // 保留两位小数
  };
}

/**
 * 格式化反馈为可读文本（用于显示给用户）
 * @param {Object} feedback - analyzeSubsectionQuality() 的返回值
 * @param {string} subsectionTitle - subsection 标题（可选）
 * @returns {string} 格式化的反馈文本
 */
function formatFeedback(feedback, subsectionTitle) {
  const lines = [];

  if (subsectionTitle) {
    lines.push(`\n📝 Feedback for: "${subsectionTitle}"`);
  }

  // 质量分
  const qualityBar = '█'.repeat(Math.round(feedback.overall_quality * 10)) +
                     '░'.repeat(10 - Math.round(feedback.overall_quality * 10));
  lines.push(`Quality: [${qualityBar}] ${(feedback.overall_quality * 100).toFixed(0)}%`);

  // 各维度评分
  lines.push(`\nScores:`);
  lines.push(`  • Clarity: ${(feedback.scores.clarity * 100).toFixed(0)}%`);
  lines.push(`  • Completeness: ${(feedback.scores.completeness * 100).toFixed(0)}%`);
  lines.push(`  • Prose Quality: ${(feedback.scores.prose_quality * 100).toFixed(0)}%`);

  // 违规
  if (feedback.violations.prose.length > 0) {
    lines.push(`\n⚠️ Prose Issues:`);
    for (const v of feedback.violations.prose) {
      lines.push(`  • ${v.rule}: ${v.message} (${v.count} found)`);
    }
  }

  if (feedback.violations.structure.length > 0) {
    lines.push(`\n⚠️ Structure Issues:`);
    for (const v of feedback.violations.structure) {
      lines.push(`  • ${v.rule}: ${v.issues.join('; ')}`);
    }
  }

  // 建议
  if (feedback.suggestions.length > 0) {
    lines.push(`\n💡 Suggestions:`);
    for (const s of feedback.suggestions) {
      lines.push(`  • ${s}`);
    }
  }

  return lines.join('\n');
}

// ---------------------------------------------------------------------------
// 导出
// ---------------------------------------------------------------------------

module.exports = {
  analyzeSubsectionQuality,
  checkProseViolations,
  checkStructureViolations,
  formatFeedback,
  PROSE_RULES,
  STRUCTURE_RULES,
  CITATION_RULES,
};
