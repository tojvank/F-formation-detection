import numpy as np

def kldiv(A, B):
    return np.sum([v for v in A * np.log2(A / B) if not np.isnan(v)])


def kl(p, q):
    ab = np.array(p.size)
    for i in range(0, p.size):
        ab[i] = p[i] * np.log(p[i] / q[i])
    return np.sum(ab)


def jsjs(p, q):
    # pp = p/norm(p,ord=1)
    # qq=q/norm(q,ord=1)
    pp = np.array(p)
    qq = np.array(q)
    mm = 0.5 * (pp + qq)
    return 0.5 * (kl(pp, mm) + kl(qq, mm))
    # return 0.5 * (entropy(pp, mm) + entropy(qq,mm))


def kl_divergence(p, q):
    somma=np.zeros(len(p))
    for i in range(0, len(p)):
        if(q[i]):
            somma[i]=sum(p[i] * np.log2(p[i] / q[i]))
        #return sum(p[i] * np.log2(p[i] / q[i]) for i in range(len(p)))
        return somma


def js_divergence(p, q):

    m=np.zeros(len(p))
    aa=[]
    for i in range(0, len(p)):
        m = 0.5 * (p[i] + q[i])
        aa = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    #m = 0.5 * (p + q)
    #return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return aa
def kkl(p,q):
    n =len(p)
    sum = 0.0
    for i in range(n):
        sum += p[i] * np.log(p[i]/q[i]) #restituisce nan
    return sum

def jjs(p, q):
    print("p+q", 1+np.log(p[0]/q[0]))
    m=0.5 * (p+q)
    left = kkl(p, m)
    right = kkl(q, m)
    print("left", left)
    print("right", right)
    return np.sort((left+right)/2)

