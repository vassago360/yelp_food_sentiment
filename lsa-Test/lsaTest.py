import numpy as np

def zeroIndex(s):
    for i, element in enumerate(s):
        if np.allclose(element, 0):
            return i
    return len(s)

np.set_printoptions(precision=2, suppress=True)

A = np.random.randn(9, 6) #A represents termsDocMatrix
U, s, V = np.linalg.svd(A, full_matrices=False)
AGeneralized = np.dot(np.dot(U[:,:-1],np.diag(s[:-1])),V[:-1,:]) #modification after removing 1 latent diagnol value from S
print("Given the random Terms-Doc matrix:\n %s \n" % str(A))
print("S (1 diagonal removed): \n %s \n" % str(np.diag(s[:-1])))
print("The matrix after LSA \"reduction\" is:\n %s \n" % str(AGeneralized))

print("\n------------------------\n")

A = np.array([[-1,-2],[3,6],[-2,-4], [5,5], [0,100]]) #Not a random termsDocMatrix
U, s, V = np.linalg.svd(A, full_matrices=False)
i = zeroIndex(s) #the ith diagonal with first 0 
s[1] = 0
print "s:", s
AGeneralized = np.dot(np.dot(U[:,:i],np.diag(s[:i])),V[:i,:])
termsLatRep = np.dot(U[:,:i],np.diag(s[:i])) #Each row is a representation of a term
docsLatRep = np.dot(np.diag(s[:i]),V[:i,:]) #Each column is a representation of a doc
print("Given the Terms-Doc matrix (not random):\n %s \n" % str(A))
print("S before and afterwards: \n %s \n\n %s\n" % (str(np.diag(s)), str(np.diag(s[:i])) ) )
print("U * S * V (without zeros): \n %s\n *\n %s\n *\n %s \n" % (str(U[:,:i]), str(np.diag(s[:i])), str(V[:i,:])) )
print("Terms latent rep (each term has %s \"latent doc dimensions\"):\n %s \n" % (str(termsLatRep.shape[1]), str(termsLatRep)) )
print("Docs latent rep (each doc has %s \"latent term dimensions\"):\n %s \n" % (str(docsLatRep.shape[0]), str(docsLatRep)) )
print("The matrix after LSA \"reduction\" is (%s diagonals removed):\n %s \n" % (str(len(s)-i), str(AGeneralized)))

#Both termsLatRep and docsLatRep represent the same thing but sort of as a mapping from inputted features to same latent features.
latDocDim = termsLatRep.shape[1]
latTermDim = docsLatRep.shape[0]
mul = np.dot(termsLatRep, docsLatRep)
#print("The %sx%s matrix of latent term-docs:\n %s \n" % (str(latTermDim), str(latDocDim), str(mul)) )




