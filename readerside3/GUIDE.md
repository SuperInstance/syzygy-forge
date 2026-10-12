# Reading guide — region questions over a tone frame (adapted from the reading experiment's v2 procedures guide)

You are shown a camera frame rendered as text: a grid of 148 columns × 80 rows of tone characters. Each character is one cell of the picture; brighter parts of the picture get "brighter" characters.

The tone ramp, darkest to brightest, with the number value of each step:

```
char:  (space)  .   :   -   =   +   *   #   %   @
value:    0     1   2   3   4   5   6   7   8   9
```

Rows are printed top to bottom (row 1 is the top of the picture). Columns run left (column 1) to right (column 148).

Regions used by the questions:

- Left half = columns 1–74. Right half = columns 75–148.
- Top half = rows 1–40. Bottom half = rows 41–80.
- Quadrants: TL = top-left, TR = top-right, BL = bottom-left, BR = bottom-right.
- Horizontal bands: top = rows 1–27, middle = rows 28–54, bottom = rows 55–80.
- An "edge" is a sharp brightness change between two neighbouring cells — a jump of about 3 or more tone steps between adjacent characters (for example `#` next to `.`, or `@` next to `:`). Smooth gradients are not edges.

## How to answer: tally first, then answer

Do not answer any question from a first impression of the whole grid. These are region questions, so work by SUMMING BY REGION:

- Work in strips of 10 rows at a time (rows 1–10, 11–20, …, 71–80).
- For brightness: in each strip, tally the total tone value of the left half and of the right half separately — count the characters at each tone step and multiply by the step value; an honest careful tally is the whole method. Write the strip tallies as Work lines.
- For edges: in each strip, count sharp steps between horizontally neighbouring cells, per column half. Write the strip counts as Work lines.
- Tally ONCE and reuse the tallies across questions: Q1 and Q2 use the quadrant totals, Q3 uses the half totals, Q4 uses the edge counts per half, Q5 uses the edge counts regrouped by band.
- A Work line starts with the word Work and the question label, like this: `Work Q1: ...`
- After a question's Work lines, write its answer line in exactly this form: `Q1: <answer>`
- Answer the questions in order, Q1 to Q5, each with its own Work lines and answer line.

## Procedure for each question

**Q1 — brightest quadrant.** Add your strip tallies into four quadrant totals (top-left, top-right, bottom-left, bottom-right). The quadrant with the LARGEST total is the answer.

**Q2 — darkest quadrant.** Use the same four quadrant totals. The quadrant with the SMALLEST total is the answer.

**Q3 — brightness centroid side.** Add all strip tallies into two totals: left half and right half. The centroid (the balance point of the brightness) normally lies on the side with the larger total. If the two totals are within a few percent of each other, check where each side's brightness sits: brightness concentrated near the outer edge pulls the centroid further than the same brightness near the centre line. Answer the side your tally supports.

**Q4 — edge-mass side.** Add your edge strip counts into two totals: left half and right half. The half with MORE edge counts is the answer.

**Q5 — edge band.** Regroup your edge strip counts by band: top = rows 1–27, middle = rows 28–54, bottom = rows 55–80. The band with the MOST edge counts is the answer.

## Standing rules

- No tally, no answer: every answer line must be preceded by its Work lines.
- If you cannot complete a tally, say so in the Work line and still give your best answer on the answer line.
- Answer lines use exactly these forms: `Q1: TL` (or TR, BL, BR) · `Q2: TL` (or TR, BL, BR) · `Q3: left` (or right) · `Q4: left` (or right) · `Q5: top` (or middle, bottom).
