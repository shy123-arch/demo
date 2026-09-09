"""Small A* planner using a conservative XY projection of scene collisions."""
import heapq
import math
import numpy as np


def obstacles(model,data,radius=.30):
    boxes=[]
    for g in range(model.ngeom):
        name=model.geom(g).name or ""
        if not name.startswith(("lab_","work_cabinet")) or "floor" in name:
            continue
        if not model.geom_contype[g] and not model.geom_conaffinity[g]:
            continue
        p=data.geom_xpos[g]
        # Static collision primitives are boxes or upright chair cylinders.
        if int(model.geom_type[g])==6:
            half=abs(data.geom_xmat[g].reshape(3,3))@model.geom_size[g]
        elif int(model.geom_type[g])==5:
            half=np.array([model.geom_size[g,0]]*2+[model.geom_size[g,1]])
        else:
            half=np.full(3,model.geom_rbound[g])
        if p[2]-half[2]>1.35 or p[2]+half[2]<.10:
            continue
        boxes.append((p[:2]-half[:2]-radius,p[:2]+half[:2]+radius))
    return boxes


def plan_path(model,data,start,goal,resolution=.15,radius=.30):
    boxes=obstacles(model,data,radius)
    origin=np.array([-3.3,-4.15])
    shape=(45,56)
    def cell(p):return tuple(np.rint((np.array(p)-origin)/resolution).astype(int))
    def world(c):return origin+np.array(c)*resolution
    def free(c):
        if not(0<=c[0]<shape[0] and 0<=c[1]<shape[1]):return False
        p=world(c)
        return not any(np.all(p>=lo) and np.all(p<=hi) for lo,hi in boxes)
    a,b=cell(start),cell(goal)
    if not free(a) or not free(b):raise ValueError("Navigation start or goal intersects an inflated obstacle")
    queue=[(0,a)]; costs={a:0}; parent={}
    while queue:
        _,c=heapq.heappop(queue)
        if c==b:break
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
            n=(c[0]+dx,c[1]+dy)
            if not free(n) or (dx and dy and (not free((c[0]+dx,c[1])) or not free((c[0],c[1]+dy)))):continue
            cost=costs[c]+math.hypot(dx,dy)
            if cost<costs.get(n,float("inf")):
                costs[n]=cost;parent[n]=c
                heapq.heappush(queue,(cost+math.dist(n,b),n))
    if b not in costs:raise RuntimeError("No collision-free patrol route")
    path=[b]
    while path[-1]!=a:path.append(parent[path[-1]])
    path.reverse()
    return [tuple(np.array(start))]+[tuple(world(c)) for c in path[1:-1]]+[tuple(np.array(goal))]
