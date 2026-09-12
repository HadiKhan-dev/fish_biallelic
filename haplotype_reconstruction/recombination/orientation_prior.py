"""Range-flip sums of the existing asymmetric orientation CTMC prior.

A summary excludes its first valid marker's initial probability. Joining two
summaries adds exactly one boundary transition (or a stationary restart).
Both all-complement variants are retained; missing markers still reset the
prior, unlike the informative-observation stream of the edge HMM.
"""
import math
import numpy as np
from numba import njit, prange


@njit(cache=True)
def _boundary(previous, current, left, right, positions, runs, rate, mean):
    stationary = rate / (rate + 1/mean)
    if left < 0 or right != left+1 or runs[left] != runs[right]:
        probability = stationary if current else 1-stationary
    else:
        changed = -math.expm1(-(rate+1/mean)*(positions[right]-positions[left]))
        enter, leave = stationary*changed, (1-stationary)*changed
        probability = ((leave if not current else 1-leave) if previous
                       else (enter if current else 1-enter))
    return math.log(probability)


@njit(cache=True)
def _value(tree, node, variant):
    score,first,last,states,lazy,roots,capacity = tree
    return score[node,variant],first[node],last[node],states[node,variant,0],states[node,variant,1]


@njit(cache=True)
def _store(tree,node,variant,value):
    score,first,last,states,lazy,roots,capacity = tree
    score[node,variant],first[node],last[node],states[node,variant,0],states[node,variant,1] = value


@njit(cache=True)
def _join(a,b,positions,runs,rate,mean):
    sa,fa,la,pa,qa = a
    sb,fb,lb,pb,qb = b
    if fa < 0:return b
    if fb < 0:return a
    return sa+sb+_boundary(qa,pb,la,fb,positions,runs,rate,mean),fa,lb,pa,qb


@njit(cache=True,parallel=True)
def _build_leaves(tree,positions,runs,jobs,rate,mean):
    for task in prange(len(jobs)):
        sample,node,left,right = jobs[task]
        for variant in range(2):
            first=-1;previous=-1;score=0.
            for site in range(left,right):
                if runs[sample,site]<0:continue
                if first<0:first=site
                else:score+=_boundary(variant,variant,previous,site,positions,runs[sample],rate,mean)
                previous=site
            _store(tree,node,variant,(score,first,previous,variant,variant))


@njit(cache=True,parallel=True)
def _build_parents(tree,positions,runs,samples,rate,mean):
    for task in prange(len(samples)):
        sample=samples[task];base=tree[5][sample]
        for local in range(tree[6][sample]-1,0,-1):
            for variant in range(2):
                _store(tree,base+local,variant,_join(
                    _value(tree,base+2*local,variant),_value(tree,base+2*local+1,variant),
                    positions,runs[sample],rate,mean))


@njit(cache=True)
def _query(tree,base,node,lo,hi,left,right,carry,positions,runs,rate,mean):
    depth=0;size=hi
    while size>1:depth+=1;size//=2
    stack=np.empty((2*depth+3,4),dtype=np.int64)
    stack[0]=(node,lo,hi,carry);top=1
    result=(0.,-1,-1,0,0)
    while top:
        top-=1;local,a,b,inherited=stack[top];absolute=base+local
        if right<=a or b<=left:value=_value(tree,absolute,inherited)
        elif left<=a and b<=right:value=_value(tree,absolute,inherited^1)
        else:
            middle=(a+b)//2;tag=inherited^int(tree[4][absolute])
            stack[top]=(2*local+1,middle,b,tag)
            stack[top+1]=(2*local,a,middle,tag);top+=2
            continue
        result=_join(result,value,positions,runs,rate,mean)
    return result


@njit(cache=True)
def _flip(tree,node):
    score,first,last,states,lazy,roots,capacity = tree
    score[node,0],score[node,1]=score[node,1],score[node,0]
    for end in range(2):
        states[node,0,end],states[node,1,end]=states[node,1,end],states[node,0,end]
    lazy[node]^=1


@njit(cache=True)
def _update(tree,base,node,lo,hi,left,right,positions,runs,rate,mean):
    depth=0;size=hi
    while size>1:depth+=1;size//=2
    stack=np.empty((3*depth+4,4),dtype=np.int64)
    stack[0]=(node,lo,hi,0);top=1
    while top:
        top-=1;local,a,b,closing=stack[top];absolute=base+local
        if right<=a or b<=left:continue
        if closing:
            for variant in range(2):
                _store(tree,absolute,variant,_join(
                    _value(tree,base+2*local,variant),_value(tree,base+2*local+1,variant),
                    positions,runs,rate,mean))
            continue
        if left<=a and b<=right:_flip(tree,absolute);continue
        if tree[4][absolute]:
            _flip(tree,base+2*local);_flip(tree,base+2*local+1)
            tree[4][absolute]=0
        middle=(a+b)//2
        stack[top]=(local,a,b,1);stack[top+1]=(2*local+1,middle,b,0)
        stack[top+2]=(2*local,a,middle,0);top+=3


@njit(cache=True)
def _score(value,positions,runs,rate,mean):
    body,first,last,phase,end = value
    if first<0:return 0.
    return body+_boundary(0,phase,-1,first,positions,runs,rate,mean)


@njit(cache=True,parallel=True)
def _scores(tree,positions,runs,jobs,rate,mean):
    result=np.empty(len(jobs))
    for task in prange(len(jobs)):
        sample,left,right=jobs[task]
        value=_query(tree,tree[5][sample],1,0,tree[6][sample],left,right,0,
                     positions,runs[sample],rate,mean)
        result[task]=_score(value,positions,runs[sample],rate,mean)
    return result


class OrientationPriorSums:
    def __init__(self,positions,runs,candidates,config,maximum_leaf_markers=4096):
        self.positions,self.runs=positions,runs
        self.args=(config.phase_error_rate,config.phase_error_mean_bp)
        sample_cuts={}
        for sample,left,right in candidates:
            sample_cuts.setdefault(sample,set()).update((left,right))
        self.samples=np.asarray(sorted(sample_cuts),dtype=np.int64)
        self.cuts={};self.job_cache={}
        roots=np.full(len(runs),-1,dtype=np.int64)
        capacity=np.zeros(len(runs),dtype=np.int64)
        jobs=[];total=0
        regular=set(range(0,len(positions),maximum_leaf_markers))|{len(positions)}
        for sample in self.samples:
            cuts=np.asarray(sorted(sample_cuts[sample]|regular),dtype=np.int64)
            self.cuts[int(sample)]=cuts
            count=len(cuts)-1;cap=1<<(count-1).bit_length()
            roots[sample]=total;capacity[sample]=cap
            jobs.extend((sample,total+cap+i,int(cuts[i]),int(cuts[i+1])) for i in range(count))
            total+=2*cap
        self.tree=(np.zeros((total,2)),np.full(total,-1,dtype=np.int64),
                   np.full(total,-1,dtype=np.int64),np.zeros((total,2,2),dtype=np.int8),
                   np.zeros(total,dtype=np.uint8),roots,capacity)
        _build_leaves(self.tree,positions,runs,np.asarray(jobs,dtype=np.int64).reshape(-1,4),*self.args)
        _build_parents(self.tree,positions,runs,self.samples,*self.args)

    def jobs(self,trials):
        rows=[]
        for sample,left,right in trials:
            key=(int(sample),int(left),int(right))
            if key not in self.job_cache:
                cuts=self.cuts[int(sample)]
                a,b=np.searchsorted(cuts,[left,right])
                assert cuts[a]==left and cuts[b]==right
                self.job_cache[key]=(sample,a,b)
            rows.append(self.job_cache[key])
        return np.asarray(rows,dtype=np.int64).reshape(-1,3)

    def score(self,jobs):
        return _scores(self.tree,self.positions,self.runs,jobs,*self.args)

    def accept(self,job):
        sample,left,right=job
        _update(self.tree,self.tree[5][sample],1,0,self.tree[6][sample],left,right,
                self.positions,self.runs[sample],*self.args)

    def current_scores(self):
        return np.asarray([_score(_value(self.tree,int(self.tree[5][s])+1,0),
                                  self.positions,self.runs[s],*self.args) for s in self.samples])
