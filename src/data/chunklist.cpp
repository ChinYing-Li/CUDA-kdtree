#include "chunklist.h"

namespace CuKee
{
ChunkList::
ChunkList()
{

}

void ChunkList::
create_chunks_from_nodes()
{

}

/*
 * Reduce node aabbs in each chunk to get per-chunk aabb
 */
void ChunkList::
chunk_reduce_aabb()
{

}

/*
 * Determine which child node the primitive belongs to.
 */
void ChunkList::
sort_prim(thrust::device_vector<bool>& res)
{

}

/*
 *
 */
void ChunkList::
clip_prim(DeviceMesh& mesh, SplitList& splitlist)
{

}

/*
 * Count primitives in each child node.
 */
void ChunkList::
count_prim_in_child()
{

}
}

