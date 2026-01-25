# Anti-AI Writing Guide: Natural Academic Prose

## Overview

AI-generated text often has distinctive patterns that can make writing feel artificial or "AI-like." This guide provides strategies to produce natural, human-like academic prose that avoids common AI writing patterns.

## Core Principles

**The Human Touch**:
- Vary sentence structure and length naturally
- Use idiomatic expressions and natural phrasing
- Include occasional imperfections or informal transitions
- Avoid over-polished, formulaic structures
- Maintain author voice and personality

## Anti-AI Prompt Template

Use this prompt when polishing or rewriting academic text:

```
Rewrite the input text to preserve its original arguments and meaning
while producing concise, natural academic prose. Apply these guidelines:

1. Sentence Length: Convert long or compound sentences into short sentences
   (aim for <20 words per sentence average)

2. Transitions: Replace all transition words and conjunctions with the most
   basic and commonly used ones (however, therefore, moreover, in addition)

3. Simplicity: Use simple expressions; avoid unnecessarily complex vocabulary
   when basic terms suffice

4. Logical Flow: Ensure logical connections between sentences are clear but
   not formulaic

5. Structure: Delete the conclusion part at the end of the text

6. Avoid AI Patterns:
   - Don't use em dashes (—) to connect clauses
   - Don't start multiple sentences with "The" or "This"
   - Vary sentence openings (prepositions, participles, conjunctions)
   - Don't use perfect symmetry in lists or examples
   - Avoid overly balanced parallel structures
   - Don't use "Furthermore," "Moreover," "Additionally" excessively
```

## Common AI Patterns to Avoid

### Pattern 1: Em Dash Overuse

**AI-like**:
```
The proposed method—which combines attention mechanisms with
transformer architectures—achieves state-of-the-art performance.
```

**Natural**:
```
The proposed method combines attention mechanisms with transformer
architectures to achieve state-of-the-art performance.
```

### Pattern 2: Formulaic Transitions

**AI-like**:
```
Furthermore, the results demonstrate significant improvement.
Moreover, the approach outperforms baselines.
Additionally, the method scales efficiently.
```

**Natural**:
```
The results demonstrate significant improvement. Our approach also
outperforms baselines while maintaining computational efficiency.
```

### Pattern 3: Perfect Parallelism

**AI-like**:
```
We tested the model on three datasets: MNIST for handwritten digits,
CIFAR for natural images, and ImageNet for object recognition.
```

**Natural**:
```
We tested the model on three datasets: MNIST, CIFAR, and ImageNet.
```

### Pattern 4: Over-structured Lists

**AI-like**:
```
Our contributions are threefold: (1) we introduce a new architecture,
(2) we demonstrate improved performance, and (3) we provide theoretical
analysis.
```

**Natural**:
```
We make three contributions. First, we introduce a new architecture.
Second, we demonstrate improved performance on standard benchmarks.
Third, we provide theoretical analysis of convergence properties.
```

### Pattern 5: Predictable Sentence Openings

**AI-like**:
```
The proposed method achieves 95% accuracy. The model consists of
three layers. The training process uses stochastic gradient descent.
```

**Natural**:
```
Our method achieves 95% accuracy. It consists of three layers trained
with stochastic gradient descent.
```

### Pattern 6: Over-qualification

**AI-like**:
```
It is important to note that the results are statistically significant.
It should be mentioned that the approach is novel.
```

**Natural**:
```
The results are statistically significant (p < 0.001). Unlike prior
work, our approach addresses X through Y.
```

## Sentence Structure Guidelines

### Vary Sentence Openings

Instead of always starting with subjects or "The," use:

**Prepositions**:
```
In contrast to previous methods...
For the task of X...
Under these conditions...
```

**Participles**:
```
Combining X with Y, we achieve...
Drawing on prior work...
Motivated by Z, we propose...
```

**Conjunctions**:
```
But the results also show...
While prior work focused on X...
And this leads to...
```

**Adverbs**:
```
Surprisingly, the method...
Typically, this approach...
Consequently, we observe...
```

### Natural Sentence Length Variation

**Mix short, medium, and long sentences**:
```
The method is fast. (3 words)

It processes input in linear time. (6 words)

By leveraging the sparse structure of the data, our algorithm achieves
computational efficiency without sacrificing accuracy, making it suitable
for real-time applications. (25 words)
```

### Connect Ideas Without Formulaic Transitions

**Instead of**:
```
The model learns features. Furthermore, it captures dependencies.
```

**Use**:
```
The model learns features while capturing dependencies.
```

**Instead of**:
```
We optimize the loss. Therefore, the model converges.
```

**Use**:
```
Optimizing the loss leads to model convergence.
```

## Vocabulary Guidelines

### Use Basic Terms When Possible

**Overly complex**:
```
The methodology utilizes sophisticated computational paradigms.
```

**Simple and clear**:
```
The method uses advanced computational approaches.
```

### Avoid Academic Jargon When Simple Words Work

**Jargon-heavy**:
```
The aforementioned aforementioned results demonstrate optimal performance.
```

**Natural**:
```
These results show the best performance.
```

### Use Field-Specific Terminology Judiciously

Only use technical terms when:
1. They are the standard in the field
2. No simpler alternative exists
3. They add precision necessary for accuracy

**Over-technical**:
```
We leverage convolutional neural network architectures with rectified
linear units to extract hierarchical feature representations.
```

**Balanced**:
```
We use CNNs with ReLU activation to extract hierarchical features.
```

## Writing Checklist

Before finalizing text, check:

**Sentence Variety**:
- [ ] Sentences vary in length (mix of 10-25 words)
- [ ] Sentence openings are diverse
- [ ] Not all sentences follow subject-verb-object pattern
- [ ] Occasional short sentences for emphasis
- [ ] Occasional longer sentences for complex ideas

**Transition Use**:
- [ ] Transitions are basic and common (however, therefore, thus)
- [ ] "Furthermore/Moreover/Additionally" used sparingly
- [ ] Connections between ideas are clear but not formulaic
- [ ] No overuse of em dashes for connecting clauses

**Vocabulary**:
- [ ] Simple words used when possible
- [ ] Technical terms necessary and field-appropriate
- [ ] No unnecessarily complex vocabulary
- [ ] Consistent terminology throughout

**Structure**:
- [ ] No perfect parallelism in every list
- [ ] Paragraphs have natural flow
- [ ] No rigid formulaic patterns
- [ ] Some sentences combine related ideas rather than separating them

**Natural Feel**:
- [ ] Text sounds like it was written by a human expert
- [ ] Author voice is present
- [ ] Occasional imperfections or stylistic choices
- [ ] Avoids "polished to perfection" feel

## Example Comparisons

### Example 1: Abstract Rewriting

**AI-like**:
```
Deep learning has emerged as a powerful tool for image classification.
Furthermore, convolutional neural networks have achieved remarkable
success. Moreover, attention mechanisms enhance model performance.
Additionally, transformer architectures show promise. Consequently,
combining these approaches yields optimal results.
```

**Natural**:
```
Deep learning has proven effective for image classification. CNNs work
particularly well, and attention mechanisms further improve performance.
Recent work shows transformers also have promise. We combine these
approaches to achieve strong results on standard benchmarks.
```

### Example 2: Methods Section

**AI-like**:
```
The proposed architecture—which consists of an encoder and a decoder—
processes input data. The encoder extracts features through convolutional
layers. The decoder reconstructs the output using transposed convolutions.
Furthermore, skip connections preserve spatial information.
```

**Natural**:
```
Our architecture has an encoder-decoder structure. The encoder extracts
features through convolutional layers, then the decoder reconstructs the
output using transposed convolutions. Skip connections preserve spatial
information throughout the network.
```

### Example 3: Results Section

**AI-like**:
```
Table 1 shows the performance comparison. Our method achieves 95.2%
accuracy. The baseline method achieves 92.1% accuracy. Thus, our method
outperforms the baseline by 3.1%. Moreover, the improvement is
statistically significant with p < 0.001.
```

**Natural**:
```
Table 1 compares performance across methods. Our approach achieves 95.2%
accuracy compared to 92.1% for the baseline, giving a 3.1% improvement.
This difference is statistically significant (p < 0.001).
```

## Quick Fixes for AI-Sounding Text

**Problem**: Text sounds too polished or formulaic

**Solutions**:
1. Combine two short sentences into one longer one
2. Break up a perfectly parallel structure
3. Replace "furthermore/moreover" with simple transitions
4. Vary sentence openings
5. Add a minor stylistic "imperfection"
6. Use simpler vocabulary where appropriate

**Example transformation**:
```
BEFORE (AI-like):
The model learns representations. Furthermore, it captures dependencies.
Moreover, it achieves state-of-the-art performance. Additionally,
it scales efficiently to large datasets.

AFTER (Natural):
The model learns representations while capturing dependencies. It
achieves state-of-the-art performance and scales efficiently to
large datasets.
```

## Common AI Phrases to Replace

| AI Phrase | Natural Alternative |
|-----------|---------------------|
| "It is important to note that" | Delete or use "Notably" |
| "It should be mentioned that" | Delete entirely |
| "Furthermore," | "Also," or combine with previous sentence |
| "Moreover," | "In addition," or restructure |
| "Additionally," | "Also," or make separate sentence |
| "Consequently," | "So," or "As a result," |
| "It is worth noting that" | Delete |
| "For this reason," | "Therefore," |
| "In order to" | "To" |
| "Due to the fact that" | "Because" |
| "In the event that" | "If" |
| "With regard to" | "About" or "For" |

## Final Tips

1. **Read aloud**: If text sounds awkward when spoken, it's too AI-like
2. **Vary rhythm**: Mix sentence lengths and structures
3. **Be concise**: Delete unnecessary words
4. **Use active voice**: Prefer "We show" over "It is shown that"
5. **Avoid perfection**: Human writing has natural variation and imperfections
6. **Keep it simple**: Use the simplest word that conveys the meaning

Remember: The goal is natural, clear academic prose—not perfectly polished
formulaic text.
