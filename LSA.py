from collections import Counter
import numpy as np
"""   This file implements the Latent Semantic Analysis    """
class LSA:
  def __init__(self):
    f=["new.txt","new2.txt","new3.txt"]
    di=[]
    map_word_index={}
    words=set()
    for i in f:
      file_f=open(i,"r")
      total_text=file_f.read()
      total_text=total_text.strip().split()
      for k in total_text:
        words.add(k)
      dic=dict(Counter(total_text))
      di+=[dic]
      file_f.close()
    i=0    
    for l in words:
      map_word_index[l]=i
      i+=1
    size_M=len(words)    
    a=np.zeros((size_M,3))
    ind=0
    for i in di:
      for j in i:
        a[map_word_index[j]][ind]=i[j]
      ind+=1
    u, s, v = np.linalg.svd(a)
    S=np.zeros((size_M,3))
    i=0
    while i<3:
      S[i][i]=s[i]
      i+=1
    s=S
    self.Map=map_word_index
    self.US=np.dot(u,s)
  def sim(self,x,y):
    return np.dot(x,y)/(pow(np.dot(x,x),0.5)*pow(np.dot(y,y),0.5))
  def totsim(self,a,S,Map,MAT):
    if(a not in Map.keys()):
        return [0]
    else:
        val=[]
        for i in S:
            if(i in Map.keys()):
                val+=[self.sim(MAT[Map[i]],MAT[Map[a]])]
        return val        
  def helper(self,S,a,MAT,Map):
    V=self.totsim(a,Map.keys(),Map,MAT)
    m=min(V)
    val=0
    for i in V:
      val+=i-m
    if(val==0):
        return 0
    return (sum(self.totsim(a,S,Map,MAT))-m)/val
  def predict(self,S,a):
    S=S.split()  
    return self.helper(S,a,self.US,self.Map)  
