#include <torch/extension.h>

#include <ATen/ATen.h>


/* std::clamp is only available since c++17 */
template <typename scalar_t>
inline constexpr const scalar_t& clamp(
    const scalar_t& v, const scalar_t& lo, const scalar_t& hi)
{
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}


/* binary search on a sorted array to find and clamp the lower bound */
template <typename scalar_t>
inline int32_t lower_bound(
        const scalar_t *data_ss,
        int32_t start,
        int32_t end,
        scalar_t val) {

    const int32_t ori_start = start;
    const int32_t upper_bound = end - start - 2;
    while (start < end) {
        int64_t mid = start + ((end - start) >> 1);
        if (!(data_ss[mid] >= val)) {
            start = mid + 1;
        }
        else {
            end = mid;
        }
    }
    return clamp(start - ori_start - 1, 0, upper_bound);
}


template <typename scalar_t>
void ialut_transform_4d_cpu_forward_impl(
        const int n,
        const scalar_t* __restrict__ data_inp,
        const scalar_t* __restrict__ data_lut,
        const int frames,
        const int height,
        const int width,
        const int stride_lut,
        const int num_channels,
        scalar_t* __restrict__ data_col) {

    const scalar_t size_bin = 1.0 / (stride_lut - 1);

    for (int index = 0; index < n; ++index) {

        /* retrieve rgbc value of the pixel */
        const scalar_t r = data_inp[index];
        const scalar_t g = data_inp[index + frames * height * width];
        const scalar_t b = data_inp[index + frames * height * width * 2];
        const scalar_t c = data_inp[index + frames * height * width * 3];

        /* retrieve index of the interpolation verticess */
        const int32_t rid = clamp((int32_t)floor(r * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t gid = clamp((int32_t)floor(g * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t bid = clamp((int32_t)floor(b * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t cid = clamp((int32_t)floor(c * (stride_lut - 1)), 0, stride_lut - 2);

        /* utility variables for indexing */
        const int stride_lut_2 = stride_lut * stride_lut;
        const int stride_lut_3 = stride_lut_2 * stride_lut;
        const int stride_lut_4 = stride_lut_3 * stride_lut;

        /* retrieve the interpolation verticess (number of 16 in case of Quadrilinear interpolation) */
        const int id0000 = (rid    ) + stride_lut * (gid    ) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid    );
        const int id1000 = (rid + 1) + stride_lut * (gid    ) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid    );
        const int id0100 = (rid    ) + stride_lut * (gid + 1) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid    );
        const int id0010 = (rid    ) + stride_lut * (gid    ) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid    );
        const int id0001 = (rid    ) + stride_lut * (gid    ) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid + 1);
        const int id1100 = (rid + 1) + stride_lut * (gid + 1) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid    );
        const int id1010 = (rid + 1) + stride_lut * (gid    ) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid    );
        const int id1001 = (rid + 1) + stride_lut * (gid    ) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid + 1);
        const int id0110 = (rid    ) + stride_lut * (gid + 1) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid    );
        const int id0101 = (rid    ) + stride_lut * (gid + 1) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid + 1);
        const int id0011 = (rid    ) + stride_lut * (gid    ) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid + 1);
        const int id1110 = (rid + 1) + stride_lut * (gid + 1) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid    );
        const int id1101 = (rid + 1) + stride_lut * (gid + 1) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid + 1);
        const int id1011 = (rid + 1) + stride_lut * (gid    ) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid + 1);
        const int id0111 = (rid    ) + stride_lut * (gid + 1) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid + 1);
        const int id1111 = (rid + 1) + stride_lut * (gid + 1) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid + 1);

        /* compute interpolation weights */
        const scalar_t rd = (r - size_bin * rid) / size_bin;
        const scalar_t gd = (g - size_bin * gid) / size_bin;
        const scalar_t bd = (b - size_bin * bid) / size_bin;
        const scalar_t cd = (c - size_bin * cid) / size_bin;

        const scalar_t w0000 = (1 - rd) * (1 - gd) * (1 - bd) * (1 - cd);
        const scalar_t w1000 = (    rd) * (1 - gd) * (1 - bd) * (1 - cd);
        const scalar_t w0100 = (1 - rd) * (    gd) * (1 - bd) * (1 - cd);
        const scalar_t w0010 = (1 - rd) * (1 - gd) * (    bd) * (1 - cd);
        const scalar_t w0001 = (1 - rd) * (1 - gd) * (1 - bd) * (    cd);
        const scalar_t w1100 = (    rd) * (    gd) * (1 - bd) * (1 - cd);
        const scalar_t w1010 = (    rd) * (1 - gd) * (    bd) * (1 - cd);
        const scalar_t w1001 = (    rd) * (1 - gd) * (1 - bd) * (    cd);
        const scalar_t w0110 = (1 - rd) * (    gd) * (    bd) * (1 - cd);
        const scalar_t w0101 = (1 - rd) * (    gd) * (1 - bd) * (    cd);
        const scalar_t w0011 = (1 - rd) * (1 - gd) * (    bd) * (    cd);
        const scalar_t w1110 = (    rd) * (    gd) * (    bd) * (1 - cd);
        const scalar_t w1101 = (    rd) * (    gd) * (1 - bd) * (    cd);
        const scalar_t w1011 = (    rd) * (1 - gd) * (    bd) * (    cd);
        const scalar_t w0111 = (1 - rd) * (    gd) * (    bd) * (    cd);
        const scalar_t w1111 = (    rd) * (    gd) * (    bd) * (    cd);

        /* Execute the interpolation */
        for (int i = 0; i < num_channels; ++i) {
            data_col[index + frames * height * width * i] =
                w0000 * data_lut[id0000 + stride_lut_4 * i] + w1000 * data_lut[id1000 + stride_lut_4 * i] +
                w0100 * data_lut[id0100 + stride_lut_4 * i] + w0010 * data_lut[id0010 + stride_lut_4 * i] +
                w0001 * data_lut[id0001 + stride_lut_4 * i] + w1100 * data_lut[id1100 + stride_lut_4 * i] +
                w1010 * data_lut[id1010 + stride_lut_4 * i] + w1001 * data_lut[id1001 + stride_lut_4 * i] +
                w0110 * data_lut[id0110 + stride_lut_4 * i] + w0101 * data_lut[id0101 + stride_lut_4 * i] +
                w0011 * data_lut[id0011 + stride_lut_4 * i] + w1110 * data_lut[id1110 + stride_lut_4 * i] +
                w1101 * data_lut[id1101 + stride_lut_4 * i] + w1011 * data_lut[id1011 + stride_lut_4 * i] +
                w0111 * data_lut[id0111 + stride_lut_4 * i] + w1111 * data_lut[id1111 + stride_lut_4 * i];
        }
    }
}


template <typename scalar_t>
void ialut_transform_4d_cpu_backward_impl(
        const int n,
        const scalar_t* __restrict__ grad_output,
        const scalar_t* __restrict__ data_inp,
        const scalar_t* __restrict__ data_lut,
        const int frames,
        const int height,
        const int width,
        const int stride_lut,
        const int num_channels,
        scalar_t* __restrict__ grad_inp,
        scalar_t* __restrict__ grad_lut) {

    const scalar_t size_bin = 1.0 / (stride_lut - 1);

    for (int index = 0; index < n; ++index) {

        /* retrieve rgbc value of the pixel */
        const scalar_t r = data_inp[index];
        const scalar_t g = data_inp[index + frames * height * width];
        const scalar_t b = data_inp[index + frames * height * width * 2];
        const scalar_t c = data_inp[index + frames * height * width * 3];

        /* retrieve index of the interpolation verticess */
        const int32_t rid = clamp((int32_t)floor(r * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t gid = clamp((int32_t)floor(g * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t bid = clamp((int32_t)floor(b * (stride_lut - 1)), 0, stride_lut - 2);
        const int32_t cid = clamp((int32_t)floor(c * (stride_lut - 1)), 0, stride_lut - 2);

        /* utility variables for indexing */
        const int stride_lut_2 = stride_lut * stride_lut;
        const int stride_lut_3 = stride_lut_2 * stride_lut;
        const int stride_lut_4 = stride_lut_3 * stride_lut;

        /* retrieve the interpolation verticess (number of 16 in case of Quadrilinear interpolation) */
        const int id0000 = (rid    ) + stride_lut * (gid    ) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid    );
        const int id1000 = (rid + 1) + stride_lut * (gid    ) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid    );
        const int id0100 = (rid    ) + stride_lut * (gid + 1) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid    );
        const int id0010 = (rid    ) + stride_lut * (gid    ) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid    );
        const int id0001 = (rid    ) + stride_lut * (gid    ) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid + 1);
        const int id1100 = (rid + 1) + stride_lut * (gid + 1) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid    );
        const int id1010 = (rid + 1) + stride_lut * (gid    ) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid    );
        const int id1001 = (rid + 1) + stride_lut * (gid    ) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid + 1);
        const int id0110 = (rid    ) + stride_lut * (gid + 1) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid    );
        const int id0101 = (rid    ) + stride_lut * (gid + 1) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid + 1);
        const int id0011 = (rid    ) + stride_lut * (gid    ) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid + 1);
        const int id1110 = (rid + 1) + stride_lut * (gid + 1) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid    );
        const int id1101 = (rid + 1) + stride_lut * (gid + 1) + stride_lut_2 * (bid    ) + stride_lut_3 * (cid + 1);
        const int id1011 = (rid + 1) + stride_lut * (gid    ) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid + 1);
        const int id0111 = (rid    ) + stride_lut * (gid + 1) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid + 1);
        const int id1111 = (rid + 1) + stride_lut * (gid + 1) + stride_lut_2 * (bid + 1) + stride_lut_3 * (cid + 1);

        /* compute interpolation weights */
        const scalar_t rd = (r - size_bin * rid) / size_bin;
        const scalar_t gd = (g - size_bin * gid) / size_bin;
        const scalar_t bd = (b - size_bin * bid) / size_bin;
        const scalar_t cd = (c - size_bin * cid) / size_bin;

        const scalar_t w0000 = (1 - rd) * (1 - gd) * (1 - bd) * (1 - cd);
        const scalar_t w1000 = (    rd) * (1 - gd) * (1 - bd) * (1 - cd);
        const scalar_t w0100 = (1 - rd) * (    gd) * (1 - bd) * (1 - cd);
        const scalar_t w0010 = (1 - rd) * (1 - gd) * (    bd) * (1 - cd);
        const scalar_t w0001 = (1 - rd) * (1 - gd) * (1 - bd) * (    cd);
        const scalar_t w1100 = (    rd) * (    gd) * (1 - bd) * (1 - cd);
        const scalar_t w1010 = (    rd) * (1 - gd) * (    bd) * (1 - cd);
        const scalar_t w1001 = (    rd) * (1 - gd) * (1 - bd) * (    cd);
        const scalar_t w0110 = (1 - rd) * (    gd) * (    bd) * (1 - cd);
        const scalar_t w0101 = (1 - rd) * (    gd) * (1 - bd) * (    cd);
        const scalar_t w0011 = (1 - rd) * (1 - gd) * (    bd) * (    cd);
        const scalar_t w1110 = (    rd) * (    gd) * (    bd) * (1 - cd);
        const scalar_t w1101 = (    rd) * (    gd) * (1 - bd) * (    cd);
        const scalar_t w1011 = (    rd) * (1 - gd) * (    bd) * (    cd);
        const scalar_t w0111 = (1 - rd) * (    gd) * (    bd) * (    cd);
        const scalar_t w1111 = (    rd) * (    gd) * (    bd) * (    cd);

        /* derivatives: w to rd */
        const scalar_t w0000_rd = - (1 - gd) * (1 - bd) * (1 - cd);
        const scalar_t w1000_rd =   (1 - gd) * (1 - bd) * (1 - cd);
        const scalar_t w0100_rd = - (    gd) * (1 - bd) * (1 - cd);
        const scalar_t w0010_rd = - (1 - gd) * (    bd) * (1 - cd);
        const scalar_t w0001_rd = - (1 - gd) * (1 - bd) * (    cd);
        const scalar_t w1100_rd =   (    gd) * (1 - bd) * (1 - cd);
        const scalar_t w1010_rd =   (1 - gd) * (    bd) * (1 - cd);
        const scalar_t w1001_rd =   (1 - gd) * (1 - bd) * (    cd);
        const scalar_t w0110_rd = - (    gd) * (    bd) * (1 - cd);
        const scalar_t w0101_rd = - (    gd) * (1 - bd) * (    cd);
        const scalar_t w0011_rd = - (1 - gd) * (    bd) * (    cd);
        const scalar_t w1110_rd =   (    gd) * (    bd) * (1 - cd);
        const scalar_t w1101_rd =   (    gd) * (1 - bd) * (    cd);
        const scalar_t w1011_rd =   (1 - gd) * (    bd) * (    cd);
        const scalar_t w0111_rd = - (    gd) * (    bd) * (    cd);
        const scalar_t w1111_rd =   (    gd) * (    bd) * (    cd);

        /* derivatives: w to gd */
        const scalar_t w0000_gd = - (1 - rd) * (1 - bd) * (1 - cd);
        const scalar_t w1000_gd = - (    rd) * (1 - bd) * (1 - cd);
        const scalar_t w0100_gd =   (1 - rd) * (1 - bd) * (1 - cd);
        const scalar_t w0010_gd = - (1 - rd) * (    bd) * (1 - cd);
        const scalar_t w0001_gd = - (1 - rd) * (1 - bd) * (    cd);
        const scalar_t w1100_gd =   (    rd) * (1 - bd) * (1 - cd);
        const scalar_t w1010_gd = - (    rd) * (    bd) * (1 - cd);
        const scalar_t w1001_gd = - (    rd) * (1 - bd) * (    cd);
        const scalar_t w0110_gd =   (1 - rd) * (    bd) * (1 - cd);
        const scalar_t w0101_gd =   (1 - rd) * (1 - bd) * (    cd);
        const scalar_t w0011_gd = - (1 - rd) * (    bd) * (    cd);
        const scalar_t w1110_gd =   (    rd) * (    bd) * (1 - cd);
        const scalar_t w1101_gd =   (    rd) * (1 - bd) * (    cd);
        const scalar_t w1011_gd = - (    rd) * (    bd) * (    cd);
        const scalar_t w0111_gd =   (1 - rd) * (    bd) * (    cd);
        const scalar_t w1111_gd =   (    rd) * (    bd) * (    cd);

        /* derivatives: w to bd */
        const scalar_t w0000_bd = - (1 - rd) * (1 - gd) * (1 - cd);
        const scalar_t w1000_bd = - (    rd) * (1 - gd) * (1 - cd);
        const scalar_t w0100_bd = - (1 - rd) * (    gd) * (1 - cd);
        const scalar_t w0010_bd =   (1 - rd) * (1 - gd) * (1 - cd);
        const scalar_t w0001_bd = - (1 - rd) * (1 - gd) * (    cd);
        const scalar_t w1100_bd = - (    rd) * (    gd) * (1 - cd);
        const scalar_t w1010_bd =   (    rd) * (1 - gd) * (1 - cd);
        const scalar_t w1001_bd = - (    rd) * (1 - gd) * (    cd);
        const scalar_t w0110_bd =   (1 - rd) * (    gd) * (1 - cd);
        const scalar_t w0101_bd = - (1 - rd) * (    gd) * (    cd);
        const scalar_t w0011_bd =   (1 - rd) * (1 - gd) * (    cd);
        const scalar_t w1110_bd =   (    rd) * (    gd) * (1 - cd);
        const scalar_t w1101_bd = - (    rd) * (    gd) * (    cd);
        const scalar_t w1011_bd =   (    rd) * (1 - gd) * (    cd);
        const scalar_t w0111_bd =   (1 - rd) * (    gd) * (    cd);
        const scalar_t w1111_bd =   (    rd) * (    gd) * (    cd);

        /* derivatives: w to cd */
        const scalar_t w0000_cd = - (1 - rd) * (1 - gd) * (1 - bd);
        const scalar_t w1000_cd = - (    rd) * (1 - gd) * (1 - bd);
        const scalar_t w0100_cd = - (1 - rd) * (    gd) * (1 - bd);
        const scalar_t w0010_cd = - (1 - rd) * (1 - gd) * (    bd);
        const scalar_t w0001_cd =   (1 - rd) * (1 - gd) * (1 - bd);
        const scalar_t w1100_cd = - (    rd) * (    gd) * (1 - bd);
        const scalar_t w1010_cd = - (    rd) * (1 - gd) * (    bd);
        const scalar_t w1001_cd =   (    rd) * (1 - gd) * (1 - bd);
        const scalar_t w0110_cd = - (1 - rd) * (    gd) * (    bd);
        const scalar_t w0101_cd =   (1 - rd) * (    gd) * (1 - bd);
        const scalar_t w0011_cd =   (1 - rd) * (1 - gd) * (    bd);
        const scalar_t w1110_cd = - (    rd) * (    gd) * (    bd);
        const scalar_t w1101_cd =   (    rd) * (    gd) * (1 - bd);
        const scalar_t w1011_cd =   (    rd) * (1 - gd) * (    bd);
        const scalar_t w0111_cd =   (1 - rd) * (    gd) * (    bd);
        const scalar_t w1111_cd =   (    rd) * (    gd) * (    bd);

        for (int i = 0; i < num_channels; ++i) {
            scalar_t grad_o_ = grad_output[index + frames * height * width * i];

            /* compute gradient of lut */
            grad_lut[id0000 + stride_lut_4 * i] += grad_o_ * w0000;
            grad_lut[id1000 + stride_lut_4 * i] += grad_o_ * w1000;
            grad_lut[id0100 + stride_lut_4 * i] += grad_o_ * w0100;
            grad_lut[id0010 + stride_lut_4 * i] += grad_o_ * w0010;
            grad_lut[id0001 + stride_lut_4 * i] += grad_o_ * w0001;
            grad_lut[id1100 + stride_lut_4 * i] += grad_o_ * w1100;
            grad_lut[id1010 + stride_lut_4 * i] += grad_o_ * w1010;
            grad_lut[id1001 + stride_lut_4 * i] += grad_o_ * w1001;
            grad_lut[id0110 + stride_lut_4 * i] += grad_o_ * w0110;
            grad_lut[id0101 + stride_lut_4 * i] += grad_o_ * w0101;
            grad_lut[id0011 + stride_lut_4 * i] += grad_o_ * w0011;
            grad_lut[id1110 + stride_lut_4 * i] += grad_o_ * w1110;
            grad_lut[id1101 + stride_lut_4 * i] += grad_o_ * w1101;
            grad_lut[id1011 + stride_lut_4 * i] += grad_o_ * w1011;
            grad_lut[id0111 + stride_lut_4 * i] += grad_o_ * w0111;
            grad_lut[id1111 + stride_lut_4 * i] += grad_o_ * w1111;

            /* compute gradient of vertices */
            scalar_t grad_d = 0;
            const scalar_t lut0000 = data_lut[id0000 + stride_lut_4 * i];
            const scalar_t lut1000 = data_lut[id1000 + stride_lut_4 * i];
            const scalar_t lut0100 = data_lut[id0100 + stride_lut_4 * i];
            const scalar_t lut0010 = data_lut[id0010 + stride_lut_4 * i];
            const scalar_t lut0001 = data_lut[id0001 + stride_lut_4 * i];
            const scalar_t lut1100 = data_lut[id1100 + stride_lut_4 * i];
            const scalar_t lut1010 = data_lut[id1010 + stride_lut_4 * i];
            const scalar_t lut1001 = data_lut[id1001 + stride_lut_4 * i];
            const scalar_t lut0110 = data_lut[id0110 + stride_lut_4 * i];
            const scalar_t lut0101 = data_lut[id0101 + stride_lut_4 * i];
            const scalar_t lut0011 = data_lut[id0011 + stride_lut_4 * i];
            const scalar_t lut1110 = data_lut[id1110 + stride_lut_4 * i];
            const scalar_t lut1101 = data_lut[id1101 + stride_lut_4 * i];
            const scalar_t lut1011 = data_lut[id1011 + stride_lut_4 * i];
            const scalar_t lut0111 = data_lut[id0111 + stride_lut_4 * i];
            const scalar_t lut1111 = data_lut[id1111 + stride_lut_4 * i];
            grad_d = grad_o_ *
                (w0000_rd * lut0000 + w1000_rd * lut1000 + w0100_rd * lut0100 + w0010_rd * lut0010 +
                 w0001_rd * lut0001 + w1100_rd * lut1100 + w1010_rd * lut1010 + w1001_rd * lut1001 +
                 w0110_rd * lut0110 + w0101_rd * lut0101 + w0011_rd * lut0011 + w1110_rd * lut1110 +
                 w1101_rd * lut1101 + w1011_rd * lut1011 + w0111_rd * lut0111 + w1111_rd * lut1111);
            // r
            grad_inp[index] += grad_d * 1 / size_bin;

            grad_d = grad_o_ *
                (w0000_gd * lut0000 + w1000_gd * lut1000 + w0100_gd * lut0100 + w0010_gd * lut0010 +
                 w0001_gd * lut0001 + w1100_gd * lut1100 + w1010_gd * lut1010 + w1001_gd * lut1001 +
                 w0110_gd * lut0110 + w0101_gd * lut0101 + w0011_gd * lut0011 + w1110_gd * lut1110 +
                 w1101_gd * lut1101 + w1011_gd * lut1011 + w0111_gd * lut0111 + w1111_gd * lut1111);
            // g
            grad_inp[index + frames * height * width] += grad_d * 1 / size_bin;

            grad_d = grad_o_ *
                (w0000_bd * lut0000 + w1000_bd * lut1000 + w0100_bd * lut0100 + w0010_bd * lut0010 +
                 w0001_bd * lut0001 + w1100_bd * lut1100 + w1010_bd * lut1010 + w1001_bd * lut1001 +
                 w0110_bd * lut0110 + w0101_bd * lut0101 + w0011_bd * lut0011 + w1110_bd * lut1110 +
                 w1101_bd * lut1101 + w1011_bd * lut1011 + w0111_bd * lut0111 + w1111_bd * lut1111);
            // b
            grad_inp[index + frames * height * width * 2] += grad_d * 1 / size_bin;

            grad_d = grad_o_ *
                (w0000_cd * lut0000 + w1000_cd * lut1000 + w0100_cd * lut0100 + w0010_cd * lut0010 +
                 w0001_cd * lut0001 + w1100_cd * lut1100 + w1010_cd * lut1010 + w1001_cd * lut1001 +
                 w0110_cd * lut0110 + w0101_cd * lut0101 + w0011_cd * lut0011 + w1110_cd * lut1110 +
                 w1101_cd * lut1101 + w1011_cd * lut1011 + w0111_cd * lut0111 + w1111_cd * lut1111);
            // c
            grad_inp[index + frames * height * width * 3] += grad_d * 1 / size_bin;
        }
    }
}


void IALUT_transform_cpu_forward(
    const torch::Tensor &input,
    const torch::Tensor &lut,
    torch::Tensor output) {

    /* retrieve some meta-information of the input tensors */
    int batch_size = input.size(0);
    int frames     = input.size(2);
    int height     = input.size(3);
    int width      = input.size(4);

    int num_channels = lut.size(1);
    int stride_lut   = lut.size(2);

    int num_kernels = frames * height * width;

    for (int elt = 0; elt < batch_size; ++elt) {
        AT_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "ialut_transform_4d_cpu_forward", ([&] {
                const scalar_t *data_inp = input[elt].data_ptr<scalar_t>();
                const scalar_t *data_lut = lut[elt].data_ptr<scalar_t>();
                scalar_t *data_col = output[elt].data_ptr<scalar_t>();

                ialut_transform_4d_cpu_forward_impl(
                    num_kernels, data_inp, data_lut, frames,
                    height, width, stride_lut, num_channels,
                    data_col);
            }));
    }
}


void IALUT_transform_cpu_backward(
    const torch::Tensor &grad_output, const torch::Tensor &input,
    const torch::Tensor &lut, torch::Tensor grad_inp, torch::Tensor grad_lut) {

    /* retrieve some meta-information of the input tensors */
    int batch_size = input.size(0);
    int frames     = input.size(2);
    int height     = input.size(3);
    int width      = input.size(4);

    int num_channels = lut.size(1);
    int stride_lut   = lut.size(2);

    int num_kernels = frames * height * width;

    for (int elt = 0; elt < batch_size; ++elt) {
        AT_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "ialut_transform_4d_cpu_backwar", ([&] {
                const scalar_t *grad_out = grad_output[elt].data_ptr<scalar_t>();
                const scalar_t *data_inp = input[elt].data_ptr<scalar_t>();
                const scalar_t *data_lut = lut[elt].data_ptr<scalar_t>();
                scalar_t *grad_inp_  = grad_inp[elt].data_ptr<scalar_t>();
                scalar_t *grad_lut_ = grad_lut[elt].data_ptr<scalar_t>();

                ialut_transform_4d_cpu_backward_impl(
                    num_kernels, grad_out, data_inp, data_lut, frames,
                    height, width, stride_lut, num_channels,
                    grad_inp_, grad_lut_);
            }));
    }
}