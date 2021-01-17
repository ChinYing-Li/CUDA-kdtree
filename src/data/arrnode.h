#pragma once

namespace CuKee
{
class ArrNode
{
public:
  ArrNode();
  void split(); // Split large node in Large Node Stage; wrapper of the CUDA kernel. __global__
  // We don't immediately "add" child node to child list; instead, we keep the SplitList to
  // be used in the next few steps

  /*
   *
  ArrSmallNode should inherit from this.
  Tentative Members
  This shiuld serve as ChunkList as well.

  vector m_primitive_indices; // indexing into Mesh; Primitives belonging to the same node are grouped together
  // TODO: How would we order parent node and children node?

We append the children node to nextlist
Sort

  Mesh* mesh (or DeviceMesh* ? )
  vector m_node_start_index; // indexing into m_primitive_indices
  vector m_node_size; // How many primitives is in the node.

CHUNK SPECIFIC
Do we really have to store chunk size? I don't think so.
bool in_chunk_mode;
How should we find the corresponding node of the chunk?
m_num_chunk_in_node
m_num_primitive_in_children; // Has size 2 * sum(m_num_chunk_in_node)
(Segmented reduction???)

For the process "sort and clip triangles to child nodes"
we need to use ArrSplitList to store where we split the large nodes
(in small node, shall we make a derived class "ArrSplitCandidate" from ArrSplitList?)

Only after we've "sort and clip triangles into child nodes" may we add child node to nextlist... I believe

Step filter_small_node
for SplitList in parallell:

*/
};
}
