#define SEED 2018

#include <cassert>
#include <cstdio>
#include <iostream>
using std::cin;
using std::cout;
using std::endl;
#define min(a, b) (((a) < (b)) ? (a) : (b))

// Define MT19937 constants (32-bit RNG)
enum {
  // Assumes W = 32 (omitting this)
  N = 624,
  M = 397,
  R = 31,
  A = 0x9908B0DF,

  F = 1812433253,

  U = 11,
  // Assumes D = 0xFFFFFFFF (omitting this)

  S = 7,
  B = 0x9D2C5680,

  T = 15,
  C = 0xEFC60000,

  L = 18,

  MASK_LOWER = (1ull << R) - 1,
  MASK_UPPER = (1ull << R)
};

static uint32_t mt[N];
static uint16_t idx;

// Re-init with a given seed
void Initialize(const uint32_t seed) {
  uint32_t i;

  mt[0] = seed;

  for (i = 1; i < N; i++) {
    mt[i] = (F * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i);
  }

  idx = N;
}

static void Twist() {
  uint32_t i, x, xA;

  for (i = 0; i < N; i++) {
    x = (mt[i] & MASK_UPPER) + (mt[(i + 1) % N] & MASK_LOWER);

    xA = x >> 1;

    if (x & 0x1) {
      xA ^= A;
    }

    mt[i] = mt[(i + M) % N] ^ xA;
  }

  idx = 0;
}

// Obtain a 32-bit random number
uint32_t ExtractU32() {
  uint32_t y;
  int i = idx;

  if (idx >= N) {
    Twist();
    i = idx;
  }

  y = mt[i];
  idx = i + 1;

  y ^= (mt[i] >> U);
  y ^= (y << S) & B;
  y ^= (y << T) & C;
  y ^= (y >> L);

  return y;
}

int main(int argc, char const *argv[]) {
  uint scan = 0;
  const uint scan_size = 60;
  uint scan_result, mid;

  cout << "Enter your group number (an integer greater or equal to 1)\n";
  int gid;
  cin >> gid;
  assert(gid >= 1 && gid <= 10);

  printf("Selected group ID:%2d\n", gid);

  printf("Row 1: index\n");
  printf("Row 2: random value\n\n");

  /* Initialize the seed */
  Initialize(SEED + 12345 * gid);

  mid = 0;

  while (mid < scan_size) {
    if (mid != 0) {
      printf("\n\n");
    }

    printf("Index %2d to %2d\n", mid + 1, mid + 10);

    for (uint i = mid; i < mid + 10; ++i) {
      printf("%-12d", i + 1);
    }

    printf("\n");

    const uint min_val = 10000;
    const uint max_val = min_val * 10 - 1;

    for (uint i = mid; i < min(scan_size, mid + 10); ++i) {
      uint rand0 = min_val + (ExtractU32() % (max_val - min_val + 1));

      if (i == 0) {
        scan_result = rand0;
      } else {
        scan_result += rand0;
      }

#ifndef SAK6ZK
      printf("%5d       ", rand0);
#else
      printf("%-7d     ", scan_result);
#endif
    }

    mid += 10;
  }

  printf("\n");

  return 0;
}
