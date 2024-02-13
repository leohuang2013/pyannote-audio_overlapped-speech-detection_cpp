
import os

f1 = open( '/tmp/cpp_before_aggregation.txt' )
c1 = f1.read()
f1.close()

f2 = open( '/tmp/py_before_aggregation.txt' )
c2s = f2.readlines()
f2.close()

c1 = c1.replace('\n', '')
tmp = c1.split(',')
c1s = []
for i in range( len( tmp )):
    if tmp[i] == '':
        continue
    c1s.append( tmp[i] )

maxd = 0.001

assert( len( c1s ) == len( c2s ))

for i in range( len( c1s )):
    if( abs(float( c1s[i] ) - float( c2s[i] )) > maxd ):
        print( f"maxi={i}, c1s={c1s[i]}, c2s={c2s[i]}" )
        break
