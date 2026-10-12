# Reading guide — region questions over a frame digest (the digest channel)

You are NOT shown the frame as a grid of characters. Instead you receive a **frame digest**: exact numbers computed from the frame's pixels by the same definitions the questions use. Your job is to read the digest and compare the numbers — the tallying has already been done, exactly, by the instrument.

The frame is a camera picture of 296 × 320 pixels, divided into a cell grid of 148 columns × 80 rows for reference. Regions used by the questions:

- Left half = columns 1–74. Right half = columns 75–148.
- Top half = rows 1–40. Bottom half = rows 41–80.
- Quadrants: TL = top-left, TR = top-right, BL = bottom-left, BR = bottom-right.
- Horizontal bands: top = rows 1–27, middle = rows 28–54, bottom = rows 55–80.
- The vertical centre line of the frame lies between columns 74 and 75, i.e. at x = 74.00 in cell units.

What the digest contains:

- **Quadrant sums** — the total brightness (sum of pixel luma, 0–255 per pixel) of each quadrant, and each quadrant's **mean** luma. All four quadrants cover the same number of pixels, so comparing sums and comparing means always agree.
- **Half sums** — total brightness of the left and right halves.
- **Total / mean / min / max luma** for the whole frame.
- **Brightness centroid** — the balance point of all the brightness, given as x and y in cell units. The x value is the one Q3 asks about: compare it with the centre line at 74.00.
- **Edge counts and edge mass.** An "edge pixel" is a pixel where the brightness changes sharply (a Sobel gradient magnitude |gx|+|gy| above 200). The **count** is how many edge pixels a region contains; the **mass** is the sum of their gradient magnitudes (how strong the edges are, in total). The questions ask which region contains *more edge activity — more sharp brightness changes* — that is the **count**. The mass is given as a cross-check; it usually agrees with the count, and where the two disagree, the count is what the question asks for.
- **Strip tallies** — the frame split into 8 strips of 10 cell rows; for each strip, the brightness sum and the edge count of each half. These are the intermediate tallies a careful reader would build strip by strip from a grid; here they are precomputed. The strip brightness tallies add up to the half sums and quadrant sums, and the strip edge counts add up to the half edge counts — you may check the region lines against them.

## How to answer: compare, then answer

Do not guess from the general size of the numbers. For each question, find the digest line named in its procedure below, write the values you are comparing as a Work line, state which is largest/smallest (or which side of the centre line the centroid falls), and only then write the answer line.

- A Work line starts with the word Work and the question label, like this: `Work Q1: ...`
- After a question's Work lines, write its answer line in exactly this form: `Q1: <answer>`
- Answer the questions in order, Q1 to Q5, each with its own Work lines and answer line.

## Procedure for each question

**Q1 — brightest quadrant.** Read the quadrant sums (or means — same order). The quadrant with the LARGEST sum is the answer.

**Q2 — darkest quadrant.** Use the same quadrant sums. The quadrant with the SMALLEST sum is the answer.

**Q3 — brightness centroid side.** Read the centroid x value. If it is less than 74.00, the centroid lies on the **left**; if greater, on the **right**. Answer that side. (Do not infer this from the half sums: the centroid is given directly, and where brightness sits inside each half matters as well as how much there is.)

**Q4 — edge side.** Read the edge counts by half. The half with the LARGER count is the answer. If the two counts were exactly equal, answer left.

**Q5 — edge band.** Read the edge counts by band. The band with the LARGEST count is the answer. (If two bands tied exactly, answer the higher one: top before middle before bottom.)

## Standing rules

- No comparison, no answer: every answer line must be preceded by its Work lines quoting the digest values you compared.
- Use only the digest numbers. Do not invent tallies the digest does not contain.
- Answer lines use exactly these forms: `Q1: TL` (or TR, BL, BR) · `Q2: TL` (or TR, BL, BR) · `Q3: left` (or right) · `Q4: left` (or right) · `Q5: top` (or middle, bottom).
