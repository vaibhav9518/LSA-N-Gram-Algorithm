from nltk.util import ngrams
import nltk
import LSA
"""   This file implements the Markov Models to predict sequential data    """
def predict(sentence,Dictionary,product,text,LSA):
   l=sentence
   alpha=0.0005
   val=[]
   di=Dictionary
   for i in di.keys():
      if(len(l.split())<len(i.split())): 
       if(i[:len(l)]==l):
           val+=[i]    
   if(len(val)==0):
        return "no suggestions"
   else:    
        o=0        
        for k in val:
          pro1=(1-alpha)*product*di[k]+alpha*LSA.predict(text,k)
          pro2=(1-alpha)*product*di[val[o]]+alpha*LSA.predict(text,val[o])
          if(pro1>pro2):
            o=val.index(k)
        return  text+" "+val[o].split()[-1]
class N_GRAM:
   def __init__(self):
     total_frequencies=0
     f=open("w2_.txt")
     l=[]
     r=f.readline()
     count=0
     while r!="":
        l+=[r]
        total_frequencies+=int(r.split()[0])
        count+=1
        r=f.readline()
     f.close()
     f=open("w3_.txt")
     r=f.readline()
     while r!="":
        count+=1
        l+=[r]
        total_frequencies+=int(r.split()[0])
        r=f.readline()
     f.close()
     di={}
     average_val=0
     average_val=total_frequencies/float(count)
     for i in l:
        i=i.split()
        di[" ".join(i[1:])]=float(i[0])/total_frequencies
     L=LSA.LSA()  
     while True:    
        text=raw_input("Enter:").lower()
        tokenize=nltk.word_tokenize(text)
        product=1
        if(len(text.split())>=3):
             od=ngrams(tokenize,3)
             od2=ngrams(tokenize,2)
             od2=[od2[0],od2[-1]]
             od=od+[od2[0]]
             for l in od:
                   if(" ".join(l) not in di.keys()):
                       product=product*average_val
                   else:
                       product=product*di[" ".join(l)]
             l=" ".join(od2[-1])
             print predict(l,di,product,text,L)
        else:
             print predict(text,di,product,text,L)
             
N=N_GRAM()   
    
