#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <string>
#include <map>


int main(int argc, char *argv[]){
  int n, m;
  std::cin >> n >> m;
  if (n <= 2) {
    return 1;
  }
  std::cout << n << " " << m << std::endl;

  // コマンドライン引数で乱数のシード値を得る
  // 指定しない場合は1
  unsigned long seed = (argc > 1 ? strtoul(argv[1], NULL, 10) : 1);
  std::default_random_engine gen(seed);
  std::uniform_int_distribution<int> disto6(0,6); // [0, 6] の一様乱数
  std::uniform_int_distribution<int> disto5(0,5); // [0, 5] の一様乱数
  std::uniform_int_distribution<int> disto4(0,4); // [0, 4] の一様乱数

  std::vector<std::string> gates(m);
  std::map<int, std::string> gate_map = {
      {0, "X"},
      {1, "Z"},
      {2, "CX"},
      {3, "CZ"},
      {4, "CCX"},
      {5, "H"}
  };
  
  int h = 0;
  for (int i = 0; i < m; i++) {
      int k = (h >= 125 ? disto4(gen) : disto5(gen));
      gates[i] = gate_map[k];
      if (k == 5) // H
          h++;
  }

  std::shuffle(gates.begin(), gates.end(), gen);
  
  std::vector<int> v(n);
  for(int i = 0; i < n; i++) v[i] = i;
  
  int j = 0;
  while (j < m) {
      std::shuffle(v.begin(), v.end(), gen);
      int i = 0;
      while(i < n && j < m) {
          if (disto6(gen) == 0) { // I
              i++;
              continue;
          }
          int k = gates[j].size();
          if (i+k > n) break;
          /**
           *  k == 1 -> H, X, Z
           *  k == 2 -> CX, CZ
           *  k == 3 -> CCX
           */
          std::cout << gates[j];
          for (int l = 0; l < k; l++, i++) {
              std::cout << " " << v[i];
          }
          std::cout << std::endl;
          j++;
      }
  }
  
  return 0;
}
