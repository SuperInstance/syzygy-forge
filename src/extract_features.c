/* extract_features.c — syzygy-forge v0 corpus extraction.
 *
 * Reads the exact NV12 stream that produced the docks-text corpus
 * (syzygy-video/work/docks_base.nv12, 296x320, 297 frames) and, for every
 * 2x4 cell, writes the feature vector the forge trains on, PLUS the fixed
 * kernel tokenizer's own per-cell decision (sel + tone ramp index) computed
 * with the kernel's own functions. The kernel repo is only #included,
 * never modified.
 *
 * Per-cell features replicate the reads syz_fused performs: the 8 luma
 * taps via syz_yuv_luma_at and the per-cell RGB from syz_yuv_ingest.
 * The fixed decision is syz_ste_select4(feat={cell_luma, green, 0, 0})
 * and syz_fused_tone(cell_luma, green) — the kernel's own code path.
 *
 * Feature layout (16 x int16, all integer; means use round-half-up):
 *   0  mean_luma      mean of the 8 block luma taps            [0,255]
 *   1  min_luma       min of the 8 taps                        [0,255]
 *   2  max_luma       max of the 8 taps                        [0,255]
 *   3  top_mean       mean of tap rows 0-1                     [0,255]
 *   4  bottom_mean    mean of tap rows 2-3                     [0,255]
 *   5  left_mean      mean of tap column 0                     [0,255]
 *   6  right_mean     mean of tap column 1                     [0,255]
 *   7  green          cell RGB green channel (fixed feat[1])   [0,255]
 *   8  red            cell RGB red channel                     [0,255]
 *   9  blue           cell RGB blue channel                    [0,255]
 *  10  cell_luma      syz_luma8(cell RGB) (fixed feat[0])      [0,255]
 *  11  green_minus_rb green - (red+blue)/2 (floor div)         [-255,255]
 *  12  grad_y         bottom_mean - top_mean                   [-255,255]
 *  13  grad_x         right_mean - left_mean                   [-255,255]
 *  14  diag_diff      mean(zigzag diag taps) - mean(anti taps) [-255,255]
 *  15  texture        mean |tap - mean_luma| (floor div)       [0,255]
 *
 * Micro profile (K=8) uses features 0-7; full profile (K=16) uses all 16.
 *
 * Output record per cell: 16 x int16 features, uint8 fixed_sel,
 * uint8 fixed_tone_idx (index into " .:-=+*#%@"). Row-major, frame-major.
 *
 * Usage: extract_features <nv12_file> <width> <height> <out_bin>
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "syz_fused.h"

#define BUFFER_SIZE (1 << 20)
static uint8_t pool[BUFFER_SIZE];
static uint8_t ybuf[296 * 320];
static uint8_t uvbuf[296 * 160];

static const char RAMP[11] = " .:-=+*#%@";

static int mean_rhu(int sum, int n) { return (sum + n / 2) / n; }

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s <nv12> <w> <h> <out_bin>\n", argv[0]);
        return 2;
    }
    const char *in_path = argv[1];
    int W = atoi(argv[2]), H = atoi(argv[3]);
    const char *out_path = argv[4];
    size_t ysz = (size_t)W * H, uvsz = (size_t)W * (H / 2);
    if (ysz > sizeof ybuf || uvsz > sizeof uvbuf) {
        fprintf(stderr, "frame too large for static buffers\n");
        return 2;
    }
    FILE *in = fopen(in_path, "rb");
    if (!in) { perror("open input"); return 1; }
    FILE *out = fopen(out_path, "wb");
    if (!out) { perror("open output"); return 1; }

    SyzArena a;
    syz_arena_init(&a, pool, sizeof pool);

    int frame = 0;
    long total_cells = 0;
    long sel_hist[4] = {0, 0, 0, 0};
    for (;;) {
        if (fread(ybuf, 1, ysz, in) != ysz) break;
        if (fread(uvbuf, 1, uvsz, in) != uvsz) {
            fprintf(stderr, "short uv read at frame %d\n", frame);
            return 1;
        }
        SyzNv12 src = { .y = ybuf, .uv = uvbuf, .w = (uint32_t)W, .h = (uint32_t)H,
                        .y_stride = (uint32_t)W, .uv_stride = (uint32_t)W };
        SyzYuvFrame f;
        if (syz_yuv_ingest(&a, &src, &f) != 0) {
            fprintf(stderr, "ingest failed at frame %d\n", frame);
            return 1;
        }
        for (uint32_t cy = 0; cy < f.rows; cy++) {
            for (uint32_t cx = 0; cx < f.cols; cx++) {
                uint32_t bx = 2u * cx, by = 4u * cy;
                size_t ci = (size_t)cy * f.cols + cx;
                const uint8_t *c = &f.rgb[ci * 3u];
                /* taps[r][col], exactly the fused loop's reads */
                int tap[4][2];
                int sum = 0, mn = 255, mx = 0;
                for (int r = 0; r < 4; r++)
                    for (int col = 0; col < 2; col++) {
                        tap[r][col] = syz_yuv_luma_at(&f, (int)bx + col, (int)by + r);
                        sum += tap[r][col];
                        if (tap[r][col] < mn) mn = tap[r][col];
                        if (tap[r][col] > mx) mx = tap[r][col];
                    }
                int mean = mean_rhu(sum, 8);
                int top = mean_rhu(tap[0][0] + tap[0][1] + tap[1][0] + tap[1][1], 4);
                int bot = mean_rhu(tap[2][0] + tap[2][1] + tap[3][0] + tap[3][1], 4);
                int lft = mean_rhu(tap[0][0] + tap[1][0] + tap[2][0] + tap[3][0], 4);
                int rgt = mean_rhu(tap[0][1] + tap[1][1] + tap[2][1] + tap[3][1], 4);
                int diag = mean_rhu(tap[0][0] + tap[1][1] + tap[2][0] + tap[3][1], 4);
                int anti = mean_rhu(tap[0][1] + tap[1][0] + tap[2][1] + tap[3][0], 4);
                int mad = 0;
                for (int r = 0; r < 4; r++)
                    for (int col = 0; col < 2; col++)
                        mad += abs(tap[r][col] - mean);
                int cell_luma = syz_luma8(c[0], c[1], c[2]);
                int16_t feat[16];
                feat[0]  = (int16_t)mean;
                feat[1]  = (int16_t)mn;
                feat[2]  = (int16_t)mx;
                feat[3]  = (int16_t)top;
                feat[4]  = (int16_t)bot;
                feat[5]  = (int16_t)lft;
                feat[6]  = (int16_t)rgt;
                feat[7]  = (int16_t)c[1];
                feat[8]  = (int16_t)c[0];
                feat[9]  = (int16_t)c[2];
                feat[10] = (int16_t)cell_luma;
                feat[11] = (int16_t)((int)c[1] - ((int)c[0] + (int)c[2]) / 2);
                feat[12] = (int16_t)(bot - top);
                feat[13] = (int16_t)(rgt - lft);
                feat[14] = (int16_t)(diag - anti);
                feat[15] = (int16_t)(mad / 8);
                /* the fixed tokenizer's own decision, via the kernel itself */
                int32_t ffeat[4] = { cell_luma, c[1], 0, 0 };
                uint32_t sel = syz_ste_select4(ffeat, SYZ_STE_BASIS);
                uint8_t tone = syz_fused_tone((uint8_t)cell_luma, c[1]);
                const char *rp = strchr(RAMP, tone);
                uint8_t tone_idx = rp ? (uint8_t)(rp - RAMP) : 0;
                fwrite(feat, sizeof(int16_t), 16, out);
                fputc((int)sel, out);
                fputc((int)tone_idx, out);
                sel_hist[sel]++;
                total_cells++;
            }
        }
        syz_arena_rollback(&a, 0);
        frame++;
    }
    fclose(in);
    fclose(out);
    printf("frames=%d cells=%ld sel_hist=[%ld %ld %ld %ld]\n",
           frame, total_cells, sel_hist[0], sel_hist[1], sel_hist[2], sel_hist[3]);
    return 0;
}
