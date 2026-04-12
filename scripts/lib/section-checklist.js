/**
 * Section Checklist for Paper Writing
 *
 * 11-point verification checklist per section type.
 * Used before advancing to Phase 3 (full lint integration).
 *
 * Phase 2 implementation: structure-focused, rule-based
 * Phase 3+ : expand with policy lint integration
 *
 * Checklist items:
 * 1. Structure: proper heading hierarchy
 * 2. Citations: has references
 * 3. Figures: figure references present
 * 4. Equations: equations explained
 * 5. Paragraphs: reasonable length (not too long)
 * 6. Clarity: no vague statements
 * 7. Transitions: smooth between paragraphs
 * 8. Evidence: claims supported
 * 9. Consistency: terminology consistent
 * 10. Prose: quality checks (from instant-feedback)
 * 11. Completeness: minimum length/paragraph count
 */

'use strict';

const instantFeedback = require('./instant-feedback.js');

// ---------------------------------------------------------------------------
// Section Types & Expected Characteristics
// ---------------------------------------------------------------------------

const SECTION_TYPES = {
  background: {
    name: 'Background & Related Work',
    minWordCount: 400,
    minParagraphs: 3,
    shouldHaveCitations: true,
    shouldHaveFigures: false,
  },
  methods: {
    name: 'Methods / Approach',
    minWordCount: 300,
    minParagraphs: 2,
    shouldHaveCitations: true,
    shouldHaveFigures: true,
  },
  experiments: {
    name: 'Experiments & Results',
    minWordCount: 400,
    minParagraphs: 4,
    shouldHaveCitations: true,
    shouldHaveFigures: true,
  },
  conclusion: {
    name: 'Conclusion',
    minWordCount: 150,
    minParagraphs: 1,
    shouldHaveCitations: false,
    shouldHaveFigures: false,
  },
};

// ---------------------------------------------------------------------------
// Checklist Functions
// ---------------------------------------------------------------------------

/**
 * Check 1: Structure (heading hierarchy)
 */
function checkStructure(texContent) {
  const headingPattern = /\\(sub)?section\{([^}]+)\}/g;
  const headings = [];
  let match;
  while ((match = headingPattern.exec(texContent)) !== null) {
    headings.push({
      level: match[1] === 'sub' ? 2 : 1,
      title: match[2],
    });
  }

  const pass = headings.length >= 1;
  return {
    item: '1. Structure',
    pass,
    details: `Found ${headings.length} heading(s)`,
    violations: pass ? [] : ['No section/subsection headings found'],
  };
}

/**
 * Check 2: Citations
 */
function checkCitations(texContent, shouldHave = true) {
  const citationPattern = /\\cite\{([^}]+)\}/g;
  const citations = [];
  let match;
  while ((match = citationPattern.exec(texContent)) !== null) {
    citations.push(match[1]);
  }

  const pass = shouldHave ? citations.length > 0 : true;
  return {
    item: '2. Citations',
    pass,
    details: `Found ${citations.length} citation(s)`,
    violations: !pass && shouldHave ? ['Section should have citations but has none'] : [],
  };
}

/**
 * Check 3: Figures
 */
function checkFigures(texContent, shouldHave = true) {
  const figurePattern = /\\label\{fig:([^}]+)\}/g;
  const refPattern = /\\ref\{fig:[^}]+\}/g;

  const figures = [];
  let match;
  while ((match = figurePattern.exec(texContent)) !== null) {
    figures.push(match[1]);
  }

  const refs = (texContent.match(refPattern) || []).length;

  const pass = shouldHave ? figures.length > 0 && refs > 0 : true;
  return {
    item: '3. Figures',
    pass,
    details: `Found ${figures.length} figure(s), ${refs} reference(s)`,
    violations: !pass && shouldHave ? ['Section should have figures and references to them'] : [],
  };
}

/**
 * Check 4: Equations (basic check)
 */
function checkEquations(texContent) {
  const eqPattern = /\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]/g;
  const equations = texContent.match(eqPattern) || [];

  // 如果有公式，应该有文字解释（简化版）
  const hasExplanation = equations.length > 0 ? /where|define|equation|formula/i.test(texContent) : true;

  const pass = equations.length === 0 || hasExplanation;
  return {
    item: '4. Equations',
    pass,
    details: `Found ${equations.length} equation(s), ${hasExplanation ? 'with' : 'without'} explanation`,
    violations: !pass ? ['Equations should be explained in surrounding text'] : [],
  };
}

/**
 * Check 5: Paragraph Length
 */
function checkParagraphLength(texContent) {
  const paragraphs = texContent.split(/\n\n+/).filter(p => p.trim().length > 50);
  const violations = [];

  for (let i = 0; i < paragraphs.length; i++) {
    const sentences = (paragraphs[i].match(/[.!?]/g) || []).length;
    if (sentences > 15) {
      violations.push(`Paragraph ${i + 1}: ${sentences} sentences (max 15)`);
    }
  }

  const pass = violations.length === 0;
  return {
    item: '5. Paragraph Length',
    pass,
    details: `${paragraphs.length} paragraphs, ${violations.length} too long`,
    violations,
  };
}

/**
 * Check 6: Clarity (no vague statements)
 */
function checkClarity(texContent) {
  const vaguePatterns = [
    /\b(unclear|confusing|complex|complicated|difficult)\b/gi,
    /\b(seems|appears|might|could|may)\s+be\s+(unclear|confusing|complicated)\b/gi,
  ];

  let matches = 0;
  for (const pattern of vaguePatterns) {
    matches += (texContent.match(pattern) || []).length;
  }

  const pass = matches === 0;
  return {
    item: '6. Clarity',
    pass,
    details: `Found ${matches} potentially unclear statement(s)`,
    violations: matches > 0 ? [`Found ${matches} vague/unclear statements`] : [],
  };
}

/**
 * Check 7: Transitions (between paragraphs)
 */
function checkTransitions(texContent) {
  const paragraphs = texContent.split(/\n\n+/).filter(p => p.trim().length > 50);
  const transitionWords = /^(However|Furthermore|Moreover|Additionally|Meanwhile|Finally|As shown|Building on|In contrast|Similarly|Next|Therefore)/i;

  const withTransitions = paragraphs.filter(p => transitionWords.test(p.trim())).length;
  const ratio = paragraphs.length > 1 ? withTransitions / paragraphs.length : 1;

  const pass = ratio >= 0.5;  // 至少 50% 的段落有过渡词
  return {
    item: '7. Transitions',
    pass,
    details: `${withTransitions}/${paragraphs.length} paragraphs have transition words`,
    violations: !pass ? ['Poor paragraph-to-paragraph transitions'] : [],
  };
}

/**
 * Check 8: Evidence (claims supported)
 */
function checkEvidence(texContent) {
  // 简化版：检查是否有足够的支持（引用、数据、图表）
  const citations = (texContent.match(/\\cite\{/g) || []).length;
  const figures = (texContent.match(/\\ref\{fig:/g) || []).length;
  const tables = (texContent.match(/\\ref\{table:/g) || []).length;

  const supportCount = citations + figures + tables;
  const pass = supportCount >= 3;  // 至少 3 个支持

  return {
    item: '8. Evidence',
    pass,
    details: `${citations} citations, ${figures} figures, ${tables} tables`,
    violations: !pass ? ['Not enough evidence/citations to support claims'] : [],
  };
}

/**
 * Check 9: Consistency (terminology)
 */
function checkConsistency(texContent) {
  // 简化版：检查是否有拼写变化（例如 "method" vs "methods"）
  // 这需要更高级的分析，在 Phase 3 expand
  const pass = true;  // Phase 2 暂时通过
  return {
    item: '9. Consistency',
    pass,
    details: 'Terminology consistency check (Phase 3 expansion)',
    violations: [],
  };
}

/**
 * Check 10: Prose Quality (integrated from instant-feedback)
 */
function checkProseQuality(texContent) {
  const feedback = instantFeedback.analyzeSubsectionQuality({ text: texContent });
  const pass = feedback.overall_quality >= 0.60;  // 60% 通过

  return {
    item: '10. Prose Quality',
    pass,
    details: `Quality score: ${(feedback.overall_quality * 100).toFixed(0)}%`,
    violations: feedback.violations.prose.length > 0
      ? feedback.violations.prose.map(v => `${v.rule}: ${v.message}`)
      : [],
  };
}

/**
 * Check 11: Completeness (word count, paragraph count)
 */
function checkCompleteness(texContent, sectionType = 'default') {
  const expectedConfig = SECTION_TYPES[sectionType] || { minWordCount: 200, minParagraphs: 1 };

  const wordCount = texContent.split(/\s+/).length;
  const paragraphs = texContent.split(/\n\n+/).filter(p => p.trim().length > 50).length;

  const pass = wordCount >= expectedConfig.minWordCount && paragraphs >= expectedConfig.minParagraphs;

  const violations = [];
  if (wordCount < expectedConfig.minWordCount) {
    violations.push(`Too brief: ${wordCount} words (min ${expectedConfig.minWordCount})`);
  }
  if (paragraphs < expectedConfig.minParagraphs) {
    violations.push(`Too few paragraphs: ${paragraphs} (min ${expectedConfig.minParagraphs})`);
  }

  return {
    item: '11. Completeness',
    pass,
    details: `${wordCount} words, ${paragraphs} paragraphs`,
    violations,
  };
}

// ---------------------------------------------------------------------------
// Main Checklist Runner
// ---------------------------------------------------------------------------

/**
 * Run full 11-point checklist on a section
 * @param {{text: string, sectionType: string, title?: string}} opts
 * @returns {{items: [], passCount: number, totalCount: number, coverage: number}}
 */
function runSectionChecklist({ text, sectionType, title } = {}) {
  if (!text || text.trim().length === 0) {
    return {
      items: [],
      passCount: 0,
      totalCount: 0,
      coverage: 0.0,
      error: 'Empty or missing text',
    };
  }

  const sectionConfig = SECTION_TYPES[sectionType] || SECTION_TYPES.background;

  const items = [
    checkStructure(text),
    checkCitations(text, sectionConfig.shouldHaveCitations),
    checkFigures(text, sectionConfig.shouldHaveFigures),
    checkEquations(text),
    checkParagraphLength(text),
    checkClarity(text),
    checkTransitions(text),
    checkEvidence(text),
    checkConsistency(text),
    checkProseQuality(text),
    checkCompleteness(text, sectionType),
  ];

  const passCount = items.filter(item => item.pass).length;
  const totalCount = items.length;
  const coverage = passCount / totalCount;

  return {
    items,
    passCount,
    totalCount,
    coverage: Math.round(coverage * 100) / 100,  // 两位小数
    title,
    sectionType,
  };
}

/**
 * Format checklist results for display
 */
function formatChecklistResults(results) {
  const lines = [];

  if (results.title) {
    lines.push(`\n📋 Checklist: "${results.title}" (${results.sectionType})`);
  }

  // 总体进度
  const passBar = '✅'.repeat(results.passCount) + '❌'.repeat(results.totalCount - results.passCount);
  lines.push(`${passBar} ${results.passCount}/${results.totalCount}`);
  lines.push(`Coverage: ${(results.coverage * 100).toFixed(0)}%\n`);

  // 逐项列出
  for (const item of results.items) {
    const icon = item.pass ? '✅' : '❌';
    lines.push(`${icon} ${item.item}: ${item.details}`);
    if (item.violations.length > 0) {
      for (const v of item.violations) {
        lines.push(`    ⚠️  ${v}`);
      }
    }
  }

  return lines.join('\n');
}

// ---------------------------------------------------------------------------
// 导出
// ---------------------------------------------------------------------------

module.exports = {
  SECTION_TYPES,
  runSectionChecklist,
  formatChecklistResults,
  // Individual checks (for testing)
  checkStructure,
  checkCitations,
  checkFigures,
  checkEquations,
  checkParagraphLength,
  checkClarity,
  checkTransitions,
  checkEvidence,
  checkConsistency,
  checkProseQuality,
  checkCompleteness,
};
