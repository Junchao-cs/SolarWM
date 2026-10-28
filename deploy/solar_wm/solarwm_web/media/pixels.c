#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

/* No fast-math, no contraction: preserve the existing three float32 rounds. */
int h3_finite(const void *p, size_t n, int bits) {
    uint64_t bad = 0;
    if (bits == 16) {
        const uint16_t *a = p;
        for (size_t i=0;i<n;i++) bad |= ((a[i]&0x7f80u)==0x7f80u);
    } else if (bits == 32) {
        const uint32_t *a = p;
        for (size_t i=0;i<n;i++) bad |= ((a[i]&0x7f800000u)==0x7f800000u);
    } else return -1;
    return bad == 0;
}

int h3_rgb_bf16(const uint16_t *p, size_t h, size_t w,
                size_t sc, size_t sy, size_t sx,
                const float *mean, const float *std, uint8_t *out) {
    for (size_t c=0;c<3;c++) for(size_t y=0;y<h;y++) for(size_t x=0;x<w;x++) {
        uint32_t b = ((uint32_t)p[c*sc+y*sy+x*sx])<<16;
        float f; memcpy(&f,&b,4);
        float v=f*std[c]; v=v+mean[c];
        if (isnan(v)) return 0;
        v = v<0 ? 0 : (v>1 ? 1 : v);
        v=v*255.0f;
        out[(y*w+x)*3+c]=(uint8_t)nearbyintf(v);
    }
    return 1;
}
