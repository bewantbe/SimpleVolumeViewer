// g++ -O2 test_dict.cpp && ./a.out

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <chrono>

using namespace std;

int main()
{
  int n = 1000000;

  // generate random numbers
  vector<int> a(n);
  for (int j = 0; j < n; j++) {
    a[j] = rand();
  }
  
  // generate random permutation
  vector<int> sid(n);
  for (int j = 0; j < n; j++)
    sid[j] = j;
  sort(sid.begin(), sid.end(), [&](int i1, int i2) { return a[i1] < a[i2]; });
  
  auto start = std::chrono::steady_clock::now();
  
  // create dict
  unordered_map<int, int> dmap(int(1.1*n));
  //unordered_map<int, int> dmap;
  //dmap.reserve(n);
  for (int j = 0; j < n; j++)
    dmap[sid[j]] = j;
  
  auto end = std::chrono::steady_clock::now();
  
  printf("hello? %d\n", sid[n-1]);
  printf("%d\n", dmap[sid[n-1]]);
  printf("bk = %d\n", dmap.bucket_count());
  printf("ldn = %f\n", dmap.max_load_factor());
  printf("mbc = %ld\n", dmap.max_bucket_count());
  
  std::chrono::duration<double> elapsed_seconds = end-start;
  printf("elapsed time: %.3f s\n", elapsed_seconds.count());

  start = std::chrono::steady_clock::now();

  vector<int> x(n);
  for (int j = 0; j < n; j++)
    x[j] = dmap[sid[j]];
  
  end = std::chrono::steady_clock::now();
  elapsed_seconds = end-start;
  
  printf("x = %ld\n", x[n-1]);
  printf("elapsed time: %.3f s\n", elapsed_seconds.count());
  
  return 0;
}
