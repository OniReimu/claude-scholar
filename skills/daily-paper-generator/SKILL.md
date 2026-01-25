---
name: daily-paper-generator
description: Use when the user asks to "generate daily paper", "search arXiv for EEG papers", "find EEG decoding papers", "review brain-computer interface papers", or wants to create paper summaries for EEG/brain decoding/speech decoding research. This skill automates searching arXiv for recent papers on EEG decoding, EEG speech decoding, or brain foundation models, reviewing paper quality, and generating structured Chinese/English summaries.
version: 0.3.0
---

# Daily Paper Generator

## Overview

Automate the workflow of discovering, reviewing, and summarizing recent research papers on arXiv related to EEG decoding, brain-computer interfaces, and neural foundation models.

**Core workflow:**
1. Search arXiv for recent papers (within 3 months) using Chrome browser
2. Retrieve paper metadata from arXiv pages
3. Evaluate paper quality using structured criteria
4. Select top 3 papers
5. Generate structured summaries with Chinese and English reviews
6. Save results as Markdown files in `daily paper/` directory

## When to Use

Use this skill when:
- User asks to "generate daily paper" or "find recent EEG papers"
- User wants to discover research on EEG decoding, speech decoding from EEG, or brain foundation models
- User needs paper reviews with both Chinese and English summaries
- User wants to track recent arXiv publications in neuro/AI intersection

## Quick Reference

| Task | Method |
|------|--------|
| Search arXiv | Use Chrome MCP tools (chrome-mcp-helper) |
| Get paper details | Navigate to arXiv pages and extract metadata |
| Evaluate quality | Use criteria in `references/quality-criteria.md` |
| Write Chinese review | Follow style in `references/writing-style.md` |
| Create output | Use template in `examples/paper-template.md` |

## Workflow

### Step 1: Search arXiv Using Chrome

**Search keywords** (see `references/keywords.md` for full list):
- EEG decoding: `EEG decoding`, `brain decoding`, `neural decoding`
- Speech decoding: `speech decoding from EEG`, `EEG speech reconstruction`
- Foundation models: `EEG foundation model`, `large EEG model`, `brain foundation model`

**Method: Use Chrome browser with arXiv search**

1. **Navigate to arXiv search** using Chrome MCP tools:
   - URL: `https://arxiv.org/search/`
   - Add search parameters: `?searchtype=all&query=KEYWORDS&abstracts=show&order=-announced_date_first`

2. **Search URL pattern**:
   ```
   https://arxiv.org/search/?searchtype=all&query=EEG+decoding&abstracts=show&order=-announced_date_first
   https://arxiv.org/search/?searchtype=all&query=EEG+foundation+model&abstracts=show&order=-announced_date_first
   ```

3. **Time filtering**: Use date filters or sort by `announced_date_first` to get recent papers

4. **Extract paper information** from search results:
   - Paper title
   - Authors
   - arXiv ID
   - Abstract preview
   - Publication date

**Example Chrome commands:**
```javascript
// Navigate to arXiv search
navigate("https://arxiv.org/search/?searchtype=all&query=EEG+decoding&abstracts=show&order=-announced_date_first")

// Get search results
getTabs()  // List tabs
screenshot()  // Capture page for analysis
```

### Step 2: Retrieve Paper Details

For each candidate paper, navigate to its arXiv abs page and extract:

**URL pattern**: `https://arxiv.org/abs/ARXIV_ID`

**Extract from page**:
- Title (from `<h1>` tag)
- Authors (from `.authors` element)
- Abstract (from `blockquote.abstract`)
- Submission date (from `.dateline`)
- arXiv ID (from URL or page)
- Categories (from `.subjects`)
- Comments (if present)

**Chrome extraction example**:
```javascript
// Navigate to paper page
navigate("https://arxiv.org/abs/2507.11783")

// Extract data
getTitle()    // Paper title
getAuthors()  // Author list
getAbstract() // Abstract text
```

### Step 3: Evaluate Paper Quality

Review each paper using the 5-dimension criteria in `references/quality-criteria.md`:

| Dimension | Weight | Key Points |
|-----------|--------|------------|
| Innovation | 30% | Novelty of contribution |
| Method Completeness | 25% | Clarity and reproducibility |
| Experimental Thoroughness | 25% | Validation depth |
| Writing Quality | 10% | Clarity of expression |
| Relevance & Impact | 10% | Domain importance |

**Scoring:** Rate each dimension 1-5, calculate weighted sum.

**Process:**
1. Screen by title/abstract for relevance
2. Navigate to full paper page for detailed review
3. Score each dimension
4. Rank by total score
5. Select top 3

### Step 4: Generate Paper Summaries

For each selected paper, create a summary following the structure in `examples/paper-template.md`:

**Required fields:**
- Title
- Authors and first author institution
- arXiv link (abs page, not PDF)
- Chinese review (~300 words)
- English review (fluent academic English)
- Key figure descriptions

**Chinese review structure** (see `references/writing-style.md`):
1. Background (1-2 sentences): Research context and importance
2. Challenges (2-3 sentences): Problems with existing methods
3. Contribution (1-2 sentences): Core contribution of this work
4. Method (2-3 sentences): Key technical details
5. Results (2-3 sentences): Main findings and metrics
6. Analysis & Limitations (1-2 sentences): Significance and limitations

**Writing style:** Follow the templates and examples in `references/writing-style.md` for consistent academic tone and structure.

### Step 5: Save Output

Create Markdown files in the `daily paper/` directory:

```
daily paper/
├── 2025-01-25-1430-paper-1.md
├── 2025-01-25-1430-paper-2.md
└── 2025-01-25-1430-paper-3.md
```

**Filename format:** `YYYY-MM-DD-HHMM-paper-N.md`

**Important:** 使用时间戳（精确到分钟）避免覆盖之前生成的文件。如果同一天多次生成，时间戳会确保每次生成都是唯一的文件名。

**Template:** Use `examples/paper-template.md` as the base format.

## Chrome Search Examples

**Example 1: Search for EEG decoding papers**
```
1. Navigate to: https://arxiv.org/search/?searchtype=all&query=EEG+decoding&abstracts=show&order=-announced_date_first
2. Screenshot the results page
3. Extract top 10 paper listings
4. Navigate to each paper's abs page for details
```

**Example 2: Search for foundation models**
```
1. Navigate to: https://arxiv.org/search/?searchtype=all&query=EEG+foundation+model&abstracts=show&order=-announced_date_first
2. Filter by date (check submission dates)
3. Select recent papers (last 3 months)
```

**Example 3: Extract paper metadata**
```
1. Navigate to: https://arxiv.org/abs/PAPER_ID
2. Extract title from <h1 class="title mathjax">
3. Extract authors from .authors a
4. Extract abstract from blockquote.abstract
5. Extract date from .dateline
```

## Additional Resources

### Reference Files

- **`references/keywords.md`** - Complete search keyword list and arXiv URL patterns
- **`references/quality-criteria.md`** - Detailed 5-dimension evaluation criteria with scoring rubrics
- **`references/writing-style.md`** - Chinese review structure, templates, and example analysis

### Example Files

- **`examples/paper-template.md`** - Output Markdown template with all required fields
- **`scripts/arxiv_search.py`** - Legacy Python script (deprecated, use Chrome instead)

### Chrome MCP Tools

Use Chrome MCP tools for browser automation:
- **Navigation**: Open arXiv search and paper pages
- **Screenshot**: Capture pages for analysis
- **Tabs**: Manage multiple arXiv pages
- **Content extraction**: Parse paper metadata from HTML

## Important Notes

1. **Time range:** Search focuses on papers from the last 3 months (check submission dates)
2. **Link format:** Use arXiv abs page links (https://arxiv.org/abs/ID), not direct PDF links
3. **Review length:** Chinese reviews should be approximately 300 words
4. **Quality focus:** Prioritize content quality (innovation, method, experiments) over quantitative metrics
5. **Bilingual output:** Both Chinese and English reviews are required for each paper
6. **Chrome required:** This workflow uses Chrome browser automation via MCP tools
