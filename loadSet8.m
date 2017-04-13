
function [ X, y ] = loadSet8();
  
load( 'SSL,set=8,data.mat' );
[ m, d0 ] = size( T );
ks = unique( T(:) );
k = length( ks );
d1 = k * d0;
X = repmat( logical(0), m, d1 );
l = 0;
for( i = 1:k )
  X( :, (l+1):(l+d0) ) = ( T == ks(i) );
  l = l + d0;
end;


