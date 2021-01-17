#pragma once

#include "src/data/splitlist.h"

namespace CuKee
{
class ArrSmallNode
{
public:
  ArrSmallNode();
  ~ArrSmallNode();
  void create_split_list(); // Algo 4, PreprocessSmallNodes(smallist: list)

private:

  ArrSplitList m_split_list;
};
}
