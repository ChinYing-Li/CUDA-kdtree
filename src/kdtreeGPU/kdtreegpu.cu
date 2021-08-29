

#include "kdtreegpu.h"

namespace CuKee
{
KDTreeGPU::KDTreeGPU(int small_thres):
  m_small_threshold(small_thres)
{

}

void KDTreeGPU
::build()
{
  /*
   * Data to be used
   * nodelist: ArrNode
   * activelist: ArrNode -> will later construct chunks from it. So let's make
   * smalllist: ArrSmallNode
   * nextlist: ArrNode
*/
}

void KDTreeGPU::
filter_small_nodes(ChunkList* nextlist)
{
   thrust::less_equal<int> le;
   NodeList* casted_nextlist = dynamic_cast<NodeList*>(nextlist);

   m_small_node_flag.clear();
   m_small_node_flag.resize(casted_nextlist->num_node());
   thrust::transform(casted_nextlist->m_node_prim_num.begin(),
                     casted_nextlist->m_node_prim_num.end(),
                     thrust::constant_iterator<int>(m_small_threshold),
                     m_small_node_flag.begin(),
                     le);
  // TODO: write kernels for removing small node in nextlist
}
}
