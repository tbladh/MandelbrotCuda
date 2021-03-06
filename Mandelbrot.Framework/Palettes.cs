﻿namespace Mandelbrot.Framework
{
    public static class Palettes
    {
        public static readonly byte[] Standard = {
        158,9,216,
        169,9,204,
        186,11,187,
        202,17,164,
        218,27,138,
        230,41,112,
        237,61,84,
        238,85,60,
        231,112,41,
        217,140,26,
        198,167,19,
        174,191,18,
        148,211,24,
        121,226,36,
        93,235,54,
        69,238,76,
        48,233,101,
        32,222,128,
        22,206,155,
        17,185,180,
        20,162,202,
        29,135,219,
        44,107,232,
        64,81,238,
        88,58,237,
        114,39,230,
        142,25,216,
        168,18,197,
        192,17,174,
        212,23,147,
        227,36,119,
        236,55,92,
        238,77,67,
        234,103,47,
        222,131,31,
        205,156,21,
        184,181,18,
        160,203,20,
        133,220,30,
        107,232,44,
        81,238,64,
        58,236,89,
        39,228,116,
        25,214,144,
        18,195,171,
        18,171,195,
        25,144,214,
        38,116,229,
        57,89,237,
        80,65,238,
        106,44,233,
        134,29,221,
        159,20,204,
        184,17,183,
        205,20,158,
        221,30,131,
        232,45,106,
        238,65,79,
        237,90,56,
        228,117,38,
        214,144,25,
        194,171,18,
        170,195,18,
        143,215,26,
        116,228,38,
        89,236,57,
        64,238,81,
        44,231,107,
        29,219,136,
        20,201,162,
        17,178,187,
        22,154,208,
        33,127,224,
        48,101,234,
        70,75,238,
        93,53,236,
        120,35,227,
        146,23,213,
        173,17,193,
        196,18,169,
        215,25,142,
        229,39,115,
        237,58,87,
        238,82,63,
        232,108,43,
        219,136,28,
        200,163,20,
        177,188,18,
        152,209,22,
        125,224,34,
        99,234,50,
        74,238,71,
        53,235,95,
        36,225,122,
        24,210,149,
        17,190,175,
        19,166,198,
        27,139,217,
        41,111,231,
        61,84,237,
        84,61,238,
        110,41,231,
        139,26,218,
        165,18,200,
        189,17,177,
        210,22,151,
        225,34,123,
        234,51,97,
        238,72,72,
        235,97,51,
        225,125,34,
        210,150,23,
        189,177,18,
        165,200,19,
        138,218,28,
        111,230,41,
        84,237,61,
        61,237,86,
        41,229,113,
        26,216,141,
        19,197,167,
        17,174,191,
        24,148,212,
        36,119,227,
        54,93,236,
        76,68,238,
        101,47,234,
        128,31,224,
        153,21,208,
        179,17,188,
        201,19,164,
        218,28,136,
        231,42,110,
        237,62,83,
        237,86,60,
        230,113,40,
        216,141,26,
        197,168,19,
        173,192,18,
        147,212,24,
        119,227,37,
        92,235,54,
        67,238,78,
        47,232,103,
        31,221,131,
        22,205,156,
        17,183,182,
        20,160,203,
        30,133,220,
        44,107,232,
        65,80,238,
        89,57,237,
        116,38,229,
        143,24,215,
        170,18,196,
        194,17,173,
        213,24,146,
        228,37,118,
        236,56,91,
        238,78,66,
        233,104,46,
        221,132,30,
        204,159,21,
        182,184,18,
        158,205,21,
        132,221,31,
        105,232,45,
        79,238,66,
        56,236,90,
        38,228,117,
        25,213,145,
        18,193,172,
        18,170,196,
        25,143,215,
        39,114,230,
        58,88,237,
        81,63,238,
        107,43,233,
        135,28,220,
        161,19,203,
        185,17,181,
        206,21,156,
        223,32,129,
        233,47,103,
        238,67,77,
        236,91,55,
        227,119,37,
        213,146,24,
        193,173,18,
        169,196,18,
        142,215,26,
        114,229,39,
        87,237,58,
        63,238,82,
        43,231,109,
        28,218,137,
        20,200,164,
        17,177,188,
        23,152,209,
        34,124,225,
        50,99,234,
        71,74,238,
        95,52,236,
        122,34,227,
        148,23,212,
        174,17,192,
        197,18,168,
        216,26,140,
        230,40,114,
        237,59,86,
        238,82,62,
        231,109,42,
        218,138,27,
        199,164,19,
        176,189,18,
        151,210,23,
        123,225,35,
        97,234,51,
        72,238,73,
        51,234,97,
        34,224,124,
        23,209,150,
        17,188,177,
        19,165,200,
        27,138,218,
        41,110,231,
        62,83,238,
        86,59,238,
        112,40,231,
        139,26,217,
        166,18,199,
        190,17,176,
        211,22,150,
        226,35,122,
        235,53,95,
        238,74,70,
        234,99,49,
        224,126,33,
        208,153,22,
        187,179,18,
        164,201,20,
        137,218,28,
        109,231,42,
        83,238,62,
        60,237,86,
        41,229,113,
        26,215,142,
        18,196,169,
        17,173,193,
        24,147,213,
        37,118,228,
        55,92,236,
        78,67,238,
        103,46,234,
        130,30,223,
        151,21,210
};

    }
}
