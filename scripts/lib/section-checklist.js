/**
 * Section Checklist for Paper Writing
 *
 * 19-point verification checklist per section type (Phase 3).
 * Combines structural checks (1-11) with policy rule integration (12-19).
 *
 * Phase 2: 11-point checklist
 * Phase 3: expanded to 19 points with LaTeX-aware detection and policy rules
 *
 * Checklist items:
 * 1. Structure: proper heading hierarchy
 * 2. Citations: has references
 * 3. Figures: figure references present
 * 4. Equations: equations explained
 * 5. Paragraphs: reasonable length (not too long)
 * 6. Clarity: no vague statements
 * 7. Transitions: smooth between paragraphs (LaTeX-aware)
 * 8. Evidence: claims supported
 * 9. Consistency: terminology consistent
 * 10. Prose: quality checks (from instant-feedback)
 * 11. Completeness: minimum length/paragraph count
 * 12. Promotional Language: no hype (policy rule)
 * 13. Copula Dodge: avoid "serves as", "marks a" (policy rule)
 * 14. Superficial -ing Suffix: avoid ", highlighting", ", emphasizing" (policy rule)
 * 15. Unicode Arrows: no mathematical arrows in prose (policy rule)
 * 16. Colon List Overuse: avoid ": (1)", ": (i)" format (policy rule)
 * 17. Despite Dismissal: avoid "Despite these challenges" (policy rule)
 * 18. Vague Attributions: avoid "Experts argue", "Industry reports" (policy rule)
 * 19. Section Count: max 6 main sections (policy rule)
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
  model: {
    name: 'Formal Model / Framework',
    minWordCount: 200,
    minParagraphs: 2,
    shouldHaveCitations: false,  // Model definitions are author's own formalization
    shouldHaveFigures: false,
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
// Utility Functions
// ---------------------------------------------------------------------------

/**
 * Strip LaTeX environments (tables, lists, code, equations) before analysis
 * to avoid false positives when checking paragraph structure
 *
 * Fixes:
 * - Avoid regex backtracking on nested environments by being more restrictive
 * - Handle escaped dollars (\$100) by using negative lookbehind
 */
function stripLatexEnvironments(tex) {
  const STRIP_ENVS = ['tabular', 'itemize', 'enumerate', 'lstlisting',
    'algorithm', 'algorithmic', 'figure', 'table',
    'equation', 'align', 'multline'];
  let result = tex;

  // Use atomic-like matching: don't allow nested structures in the middle
  // Instead of [\\s\\S]*?, use a more restrictive pattern that counts braces
  for (const env of STRIP_ENVS) {
    // Match: \begin{env...} ... \end{env}
    // Middle part: [^\\]* (no backslash) or \\. (escaped char) or \\{...} (nested brace)
    // This avoids catastrophic backtracking while still handling most LaTeX
    const pattern = `\\\\begin\\{${env}[^}]*\\}(?:[^\\\\]|\\\\.|\\\\\\{[^}]*\\})*?\\\\end\\{${env}\\}`;
    const re = new RegExp(pattern, 'g');
    result = result.replace(re, '');
  }

  // Strip displayed math: $$...$$ and \[...\]
  result = result.replace(/\$\$[\s\S]*?\$\$/g, '');
  result = result.replace(/\\\[[\s\S]*?\\\]/g, '');

  // Strip inline math with proper escaped-dollar handling
  // Use negative lookbehind to avoid matching \$100
  // Pattern: (?<!\\)\$ ... (?<!\\)\$
  // This matches $ that is NOT preceded by \ (i.e., not \$)
  try {
    // Lookbehind support (ES2018+)
    result = result.replace(/(?<!\\)\$[^$\n]*?(?<!\\)\$/g, '');
  } catch {
    // Fallback for older engines: simple pattern without escaped-dollar handling
    // Accept this limitation as edge case
    result = result.replace(/\$[^$\n]+\$/g, '');
  }

  return result;
}

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
  // Strip LaTeX environments first to avoid counting table/list content
  const cleanedContent = stripLatexEnvironments(texContent);
  const paragraphs = cleanedContent.split(/\n\n+/).filter(p => p.trim().length > 50);
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
 * LaTeX-aware: detects English transition words, LaTeX structure signals
 */
function checkTransitions(texContent) {
  const cleanedContent = stripLatexEnvironments(texContent);
  const paragraphs = cleanedContent.split(/\n\n+/).filter(p => p.trim().length > 50);

  // English transition words
  const englishTransition = /^(However|Furthermore|Moreover|Additionally|Meanwhile|Finally|As shown|Building on|In contrast|Similarly|Next|Therefore|These|This|Such)/i;

  // LaTeX structural signals (subsection, labeled paragraphs, etc.)
  const latexTransitionPatterns = [
    /\\subsection\{/,                        // subsection heading
    /\\smallskip\\noindent\\textbf\{/,      // named paragraph (boldface)
    /\\paragraph\{/,                         // \paragraph command
    /\\noindent\s*\\textbf\{/,              // noindent + boldface
    /Table[~\s]\\ref\{|Fig[.~\s]\\ref\{/,  // Figure/Table reference intro
    /\\S\\ref\{sec:/,                       // Section cross-reference
  ];

  const withTransitions = paragraphs.filter(p => {
    const trimmed = p.trim();
    return englishTransition.test(trimmed) ||
           latexTransitionPatterns.some(pat => pat.test(trimmed));
  }).length;

  const ratio = paragraphs.length > 1 ? withTransitions / paragraphs.length : 1;

  // Threshold lowered from 0.5 to 0.3 for LaTeX papers (subsection already handles structure)
  const pass = ratio >= 0.3;
  return {
    item: '7. Transitions',
    pass,
    details: `${withTransitions}/${paragraphs.length} paragraphs have transition signals`,
    violations: !pass ? ['Weak paragraph-to-paragraph transitions or structure signals'] : [],
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

/**
 * Check 12: Promotional Language (policy rule: PROSE.PROMOTIONAL_LANGUAGE)
 */
function checkPromotionalLanguage(texContent) {
  const pattern = /\b(exciting|remarkable|revolutionary|groundbreaking|dramatically|game-changing|cutting-edge|unprecedented|transformative)\b/gi;
  const matches = texContent.match(pattern) || [];
  const pass = matches.length === 0;
  return {
    item: '12. Promotional Language',
    pass,
    details: `Found ${matches.length} promotional word(s)`,
    violations: pass ? [] : [`Promotional language (${matches.slice(0, 3).join(', ')}...)`],
  };
}

/**
 * Check 13: Copula Construction (policy rule: PROSE.COPULA_DODGE)
 */
function checkCopulaConstruction(texContent) {
  const pattern = /\b(serves as|stands as|marks a|represents a)\b/gi;
  const matches = texContent.match(pattern) || [];
  const pass = matches.length === 0;
  return {
    item: '13. Copula Dodge',
    pass,
    details: `Found ${matches.length} copula phrase(s)`,
    violations: pass ? [] : ['Avoid "serves as", "stands as", "marks a", "represents a"'],
  };
}

/**
 * Check 14: Superficial -ing Suffix (policy rule: PROSE.SUPERFICIAL_ING_SUFFIX)
 */
function checkIngSuffix(texContent) {
  const pattern = /,\s*(highlighting|underscoring|emphasizing|showcasing|reflecting|symbolizing|contributing to|fostering|cultivating|encompassing)\s/gi;
  const matches = texContent.match(pattern) || [];
  const pass = matches.length === 0;
  return {
    item: '14. Superficial -ing Suffix',
    pass,
    details: `Found ${matches.length} superficial ending(s)`,
    violations: pass ? [] : ['Avoid comma-ing constructs (e.g., ", highlighting", ", emphasizing")'],
  };
}

/**
 * Check 15: Unicode Arrows (policy rule: PROSE.UNICODE_ARROWS)
 */
function checkUnicodeArrows(texContent) {
  const pattern = /[→←↔⇒⇐⇔➜➔]/g;
  const matches = texContent.match(pattern) || [];
  const pass = matches.length === 0;
  return {
    item: '15. Unicode Arrows',
    pass,
    details: `Found ${matches.length} Unicode arrow(s)`,
    violations: pass ? [] : ['Use LaTeX math arrows or ASCII, not Unicode arrows'],
  };
}

/**
 * Check 16: Colon List Overuse (policy rule: PROSE.COLON_LIST_OVERUSE)
 */
function checkColonListOveruse(texContent) {
  const patterns = [
    /:\s*\(1\)/g,
    /:\s*\(i\)/g,
    /:\s*1\)/g,
  ];
  const allMatches = [];
  for (const p of patterns) {
    allMatches.push(...(texContent.match(p) || []));
  }
  const pass = allMatches.length === 0;
  return {
    item: '16. Colon List Overuse',
    pass,
    details: `Found ${allMatches.length} colon-list(s)`,
    violations: pass ? [] : ['Avoid ": (1)", ": (i)", ": 1)" numbered list format'],
  };
}

/**
 * Check 17: Despite Dismissal (policy rule: PROSE.DESPITE_DISMISSAL)
 */
function checkDespiteDismissal(texContent) {
  const pattern = /\b[Dd]espite (these|its|their|such) (challenges|limitations|drawbacks|shortcomings)\b/gi;
  const matches = texContent.match(pattern) || [];
  const pass = matches.length === 0;
  return {
    item: '17. Despite Dismissal',
    pass,
    details: `Found ${matches.length} dismissive despite-phrase(s)`,
    violations: pass ? [] : ['Avoid "Despite these challenges/limitations" (weak transition)'],
  };
}

/**
 * Check 18: Vague Attributions (policy rule: PROSE.VAGUE_ATTRIBUTIONS)
 */
function checkVagueAttributions(texContent) {
  const patterns = [
    /\b([Ee]xperts|[Oo]bservers|[Rr]esearchers|[Ss]cholars)\s+(argue|believe|suggest|note|have cited|have noted|contend|maintain)\b/gi,
    /\b([Ii]ndustry|[Rr]ecent) reports? (suggest|indicate|show)\b/gi,
  ];
  const allMatches = [];
  for (const p of patterns) {
    allMatches.push(...(texContent.match(p) || []));
  }
  const pass = allMatches.length === 0;
  return {
    item: '18. Vague Attributions',
    pass,
    details: `Found ${allMatches.length} vague attribution(s)`,
    violations: pass ? [] : ['Avoid unattributed "Experts/Researchers argue" or "Industry reports"'],
  };
}

/**
 * Check 19: Section Count (policy rule: PAPER.SECTION_HEADINGS_MAX_6)
 * Only checks full paper, not individual sections
 */
function checkSectionCount(texContent) {
  const sections = (texContent.match(/^\\section\{/gm) || []).length;
  const pass = sections <= 6;
  return {
    item: '19. Section Count',
    pass,
    details: `Found ${sections} main section(s) (max 6)`,
    violations: pass ? [] : [`Too many sections: ${sections} (max 6 recommended)`],
  };
}

// ---------------------------------------------------------------------------
// Main Checklist Runner
// ---------------------------------------------------------------------------

/**
 * Run full 19-point checklist on a section (Phase 3)
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
    // Phase 3 policy rules
    checkPromotionalLanguage(text),
    checkCopulaConstruction(text),
    checkIngSuffix(text),
    checkUnicodeArrows(text),
    checkColonListOveruse(text),
    checkDespiteDismissal(text),
    checkVagueAttributions(text),
    checkSectionCount(text),  // Only meaningful for full paper
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
  stripLatexEnvironments,
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
  // Phase 3 policy rule checks
  checkPromotionalLanguage,
  checkCopulaConstruction,
  checkIngSuffix,
  checkUnicodeArrows,
  checkColonListOveruse,
  checkDespiteDismissal,
  checkVagueAttributions,
  checkSectionCount,
};
