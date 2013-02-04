"""Is your matrix feeling a little singular today? This module can help!"""

__author__ = "Gabriel Balaban"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

import numpy as np
from dolfin import near
from instant import inline_with_numpy
 
class MatrixDoctor(object):
    """A Class for diagnosing sick(singular) matricies"""
    def __init__(self,M,zTOL = 1.0e-12):
        self.M = M
        #Tolerance for a number being close to another.
        self.zTOL = zTOL
        self.build_cfuntions()

    def build_cfuntions(self):
        c_code__isnotzero = """
            bool isnotzero(int row_n, double *row, double zTOL) {
                for ( int i=0; i<row_n; i++ )
                    {if (row[i] > zTOL || -row[i] > zTOL) return true;}
                return false;
            }
            """

        args = [["row_n", "row"]]
        self.c_isnotzero = inline_with_numpy(c_code__isnotzero, arrays=args)

    def diagnose(self):
        """Diagnose the matrix M"""
        print "Matrix Diagnosis zero and lindep tolerance = ",self.zTOL
        print "Matrix Determinant"
        print  np.linalg.det(self.M)
        print
        #diagnose rows
        self.__diagnose_row_or_col(self.M,"rows")
        print
        #Create a contiguous view of the transpose matrix
        Mt = np.array(self.M.transpose(), copy=True, order='C')
        self.__diagnose_row_or_col(Mt,"columns")

    def __diagnose_row_or_col(self,M,rowcolstr):
        """Checks the rows or columns of a matrix for linear dependance and equality to 0"""
        print " ".join(["Matrix", rowcolstr, "diagnosis"]) 
        zerorows = []
        lindep = []
        #Dictionary containing original row number as key and row as value.
        Mdic = dict(zip(range(len(M)),M))
        
        #First look for zero rows and remove them
        for key in Mdic.keys(): 
            if not self.__isnotzero(Mdic[key]):
                zerorows.append(key)
                del Mdic[key]
        #Now check the remaining rows for linear dependance
        while Mdic != {}:
            key1 = Mdic.keys()[0]
            lindeploc = [key1]
            for key2 in Mdic.keys()[1:]:
                #If the rows are dependant
                if not self.__islinindep(Mdic[key1],Mdic[key2]):
                    lindeploc += [key2]
                    del Mdic[key2]
            #Delete the row we have checked for ld
            del Mdic[key1]
            if lindeploc != [key1]:    
                lindep += [lindeploc]
        #Output the result
        if zerorows == [] and lindep == []:
            print "".join(["No lindep or 0 ",rowcolstr])               
        elif zerorows != []:            
            print " ".join(["found",str(len(zerorows)), "zero", rowcolstr])
            print zerorows
        elif lindep != []:
            print " ".join(["found",str(len(lindep)), "sets of linearly dependant", rowcolstr])
            print lindep

    def __isnotzero(self,row):
        #return self.c_isnotzero(row,self.zTOL)
        for elem in row:
            if not abs(elem) < self.zTOL:
                return True
                
    def __islinindep(self,r1,r2):
        #Check to see if row r1 (almost) a scalar multiple of r2
        #Get rid of adjacent 0's
        pairedrows = [tup for tup in zip(r1,r2) if not (abs(tup[0]) < self.zTOL and abs(tup[1]) < self.zTOL)] 
        #Check the first entry. If one of the entries is zero, the other is not by
        #the construction of pairedrows.
        if abs(pairedrows[0][1]) < self.zTOL:
            return True
        else:
            quot = pairedrows[0][0] / pairedrows[0][1]
        #Compare quotients of remaining entries to quot   
        for e1,e2 in pairedrows[1:]:
            if abs(e2) < self.zTOL or abs(quot -e1/e2) > self.zTOL:
                return True


#Test Case        
if __name__ == "__main__":
    M = np.array([[0.0000001,0.0,0.0],[2.0,2.0,2.0],[1.0,2.0,2.00000001]], dtype = "float64")
    print M
    md = MatrixDoctor(M,zTOL = 1.0e-12)
    md.diagnose()

#Post mortem debugging
#    try:
#        yourcode
#    except:
#   import pdb,sys
#    e ,m, tb = sys.exec_info()
#    pdb.post_mortem(tb)
        
##
##__isnotzero_ccode = """
##bool __isnotzero(int n1, double* array1, double* zTOL)
##{
##
##    return false; 
##}
##"""
####
####    for (int i=0; i<n1; i++)
####    {  
####        if (array1[i] > zTOL || -array1[i] > zTOL) return true; 
####    }
##__isnotzero = inline_with_numpy(__isnotzero_ccode,arrays =[['n1', 'array1','zTOL']])
##print __isnotzero
##foo = np.array([1.0,2.3,3.1,4.6])
##print __isnotzero(len(foo),foo, 0.001)
