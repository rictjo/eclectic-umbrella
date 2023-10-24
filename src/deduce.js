"""
Copyright 2023 RICHARD TJÖRNHAMMAR

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

function determineMeanAndStddev( data ) {
  const dataMean		= data.mean(0);
  const diffFromMean		= data.sub(dataMean);
  const squaredDiffFromMean	= diffFromMean.square();
  const variance		= squaredDiffFromMean.mean(0);
  const dataStd			= variance.sqrt();
  return {dataMean, dataStd};
}
// export
function standardizeTensor(data, dataMean, dataStd) {
  return data.sub(dataMean).div(dataStd);
}

/*
MY IMPETUOUS-GFA REPO CODES
https://github.com/richardtjornhammar/impetuous/blob/master/src/impetuous/reducer.py
*/

function index_value(C,i,j) {
  let CS	= tf.max( C.slice([i,j],[1,1]) );
  // LOOKS STRANGE BUT ONE LESS DATASYNC
  // console.log( C.arraySync()[i][j]==tf.max(CS).dataSync() ); // ARGH...
  return ( CS );
}

function kth_householder( A , k ) {
  let [Pk,Ak,Qk] = tf.tidy( () => { // TIDY LINE NEEDED?
  // THE ACUTAL DECOMPOSITION
  var B		= A ;
  var k0	= k ;
  var k1	= k+1 ;
  var Op5	= tf.scalar(0.5);
  let s_	= index_value( B , k1 , k0 ) ;
  var alpha	= tf.scalar( 2 ).mul( s_.less(tf.scalar(0.0)).toInt() ).sub(1);
  alpha = alpha.mul( tf.sqrt( tf.sum(B.transpose().slice([k0,k1],[1,-1]).square()) ) ) ;
  var r = tf.sqrt( Op5.mul( alpha.square().sub( alpha.mul(index_value(B,k1,k0)) ) ) );
  //		THIS COMING LINE NEEDS RETHINKING <<SYNC>>
  var d = tf.max( B.shape[0] ).sub(k1+1).reshape([1]) .dataSync()	// ARGH...
  var v_ = tf.zeros([k1]) ;
  v_ = tf.concat( [v_ , tf.reshape( index_value(B,k1,k0).sub(alpha).mul(Op5).div(r) , [1] ) ], axis=0 );
  v_ = tf.concat( [v_,tf.reshape( Op5.div(r).mul(B.slice([k1+1,k0],[-1,1])), d ) ] , axis=0 )
  var Pk = tf.eye( B.shape[0] ).sub( tf.scalar(2).mul( tf.outerProduct(v_,v_.transpose()) ) )
  var Qk = Pk
  if ( B.shape[0] != B.shape[1] ) {
     alpha	= tf.scalar( 2 * ( + index_value(B,k0,k1) < 0 ) - 1 ) ;
     alpha	= alpha.mul( tf.sqrt( tf.sum(B.slice([k0,k1],[1,-1]).square()) ) );
     r = tf.sqrt( Op5.mul( alpha.square().sub( alpha.mul(index_value(B,k0,k1)) ) ) );
     //            THIS COMING LINE NEEDS RETHINKING <<SYNC>> EQUIVALENT WITH ABOVE
     var p  = tf.reshape( tf.max( B.shape[1] ).sub(k1+1) , [1] ).dataSync();	// ARGH...
     var w_ = tf.zeros([k1]) ;
     w_ = tf.concat( [w_ , tf.reshape( index_value(B,k0,k1).sub(alpha).mul(Op5).div(r) , [1] ) ] , axis=0 );
     w_ = tf.concat( [w_ , tf.reshape( Op5.div(r).mul(B.slice([k0,k1+1],[1,-1]))    , p   ) ] )
     Qk = tf.eye( B.shape[1] ).sub( tf.scalar(2).mul( tf.outerProduct(w_,w_.transpose()) ) )
  }
  var Ak = tf.dot( tf.dot( Pk,B ),Qk );
  return [Pk,Ak,Qk];

  }); // EXIT TIDY
  //
  // NOT WITH ASYNC
  return [Pk,Ak,Qk] ;
}

function rich_rot( a , b ) { // TRY TO BUILD A TIDY FUNCTION
    let R = tf.tidy( () => {
        var c = tf.scalar(0);
        var s = tf.scalar(0);
        var r = tf.scalar(0);
        if ( !( a==0 & b==0 ) ) {
            r = tf.sqrt( a.mul(a) .add( b.mul(b) ) ) ;
            if ( a .equal( 0 ) ) {
                s = r.div( b ) ;
            } else {
                s = b.div( r ) ;
                c = r.sub( s.mul(b) ) .div(a) ;
            }
        }
        let cds = c.dataSync()[0]		// ARGH...
        let nsd = s.mul(-1).dataSync()[0]	// ARGH...
        let sds = s.dataSync()[0]		// ARGH...
        let R = tf.tensor2d( [[cds,sds],[nsd,cds]]) ;
        return R
    });
    return R ;
}

// export
// async
function householder_reduction ( M ) {
    let [P,A_,QT] = tf.tidy( () => { // TIDIER MEM ?
    // THE ACTUAL FUNCTION
    const A = tf.tensor2d(M) ;
    const nlim =  tf.reshape( tf.max( A.shape[1] ) .sub(1) , [1] ).dataSync(); // ARGH...
    if ( A.shape[0] < 2 )  {
        return ( A ) ;
    }
    let [P0,A0,Q0] = kth_householder( A , k=0 );
    // console.log('ZEROTH:',A0.dataSync());
    if ( A.shape[0] == 2 ) {
        return ( [P0,A0,Q0] );
    }
    for (var k=1 ; k<nlim ; k++ ) {
        let [P1, A1, Q1] = kth_householder( A0 , k=k )
        A0 = A1
        P0 = tf.dot( P0 , P1 );
        Q0 = tf.dot( Q0 , Q1 );
    }
    var P  = P0 ;
    var A_ = A0 ;
    var QT = Q0.transpose() ;
    return ( [P,A_,QT] );
    });
    // TIDY RETURNED
    return ( [P,A_,QT] );
}

function listToMatrix( a ) {
     A = a.map( (_,i) =>
         a.map( (v,j) =>
            i==j ? v: 0 ));
   return ( A );
}

function matrixToVector(M,ishift=0) {
    // TYPE CHECKING NEEDED. JS WHAT CAN YOU DO...
    A = tf.tensor2d(M);
    var n_ = tf.reshape( tf.min( A.shape ) , [1] ).dataSync();
    if ( ishift < 0 ) {
        A=A.transpose();
        ishift=ishift*-1
    }
    let d = A.arraySync().map( (v,i) => ishift==0 ?
		v[ i , i ] :
		v[ i , i + ishift<n_ ? i+ishift :n_-1 ] );
    return ( tf.tensor1d(d) .slice(0,n_-ishift) )
}

function tf_sgn( a ) {
   s = a.dataSync().map( ( v )  => v>=0?1:-1 )
   return ( tf.tensor1d(s) )
}

function fat_tridiagonal ( sub , main , sup ) {
   // sub , main , sup are tf.tensor1d
   sup = sup.arraySync();
   sub = sub.arraySync();
   const n_ = tf.reshape( tf.min( main.shape ) , [1] ).dataSync();
   main = main.arraySync();
   FAT = main.map( (w,i) =>
         main.map( (v,j) =>
           i==j ? w : j == i+1 & j<n_ ? sup[j-1]  : j == i-1 & j+1>0? sub[j] :0  )) ;
   return ( tf.tensor2d(FAT) );
}

function diagonalize_tridiagonal( T ) {
   if ( true ) {
     console.log('BREAK')
     const FT = [       [22         ,  -15.8338556, 0          , 0          , 0         ],
                        [ -15.8338556, 11.9500151 , -4.1214752 , 0          , 0         ],
                        [0          , -4.1214752 , 9.4409418  , -15.3210535, 0         ],
                        [0          , 0          , -15.3210535, -11.0633421, -0.0000023],
                        [0          , 0          , 0          , -0.0000023 ,  -6e-7     ]
     ]
     console.log( 'FT:' , FT )
     console.log( "TEST FT>" )
     let [ xx , yy , zz ] = householder_reduction( FT )
     yy.print()
     console.log( "BROKE" )
   }
   T = tf.tensor2d(T);
   let tridiagonal = tf.linalg.bandPart( T , 1 , 1 ); // CLEANED T
   //
   ci = matrixToVector( T.arraySync() , ishift =-1 );
   ai = matrixToVector( T.arraySync() , ishift = 0 );
   bi = matrixToVector( T.arraySync() , ishift = 1 );

   subsup_diagonals = tf.sqrt( tf.abs(bi.mul(ci)) ) .mul( tf_sgn( bi ) ) .mul( tf_sgn( ci ))
   //
   console.log('HERE AGAIN');
   //
   // FTRI = fat_tridiagonal( subsup_diagonals , ai , subsup_diagonals ) ;
   FTRI = fat_tridiagonal( ci , ai , bi ) ;
   FTRI .print()
   tridiagonal.print();
   subsup_diagonals.print();
   //
   /* THIS WILL CREATE NANS
   let [ aa , bb , cc ] = householder_reduction( tridiagonal.arraySync() )
   aa	.print()
   bb	.print()
   cc	.print()
   */
   let y = tf.linalg.gramSchmidt( FTRI );
   console.log('ADDED')
   y.print()
   let [qt,rt] = tf.linalg.qr( tridiagonal ) ; // FTRI )
   qt.print()
   tf.dot(  y,FTRI ).print()
   rt.print()
   console.log( 'BACK AGAIN' )
}

function diagonalize_2b2( A ) {
   return(tf.scalar(0));
};

async function run() {
  /*
     HOUSEHOLDER, SVD, PCA
  */
   const M = [
     [ 22, 10,  2,   3,  7],
     [ 14,  7, 10,   0,  8],
     [ -1, 13, -1, -11,  3],
     [ -3, -2, 13,  -2,  4],
     [  9,  8,  1,  -2,  4],
     [  9,  1, -7,   5, -1],
     [  2, -6,  6,   5,  1],
     [  4,  5,  0,  -2,  2]
   ]
   // console.log( SVDJS.SVD( M ) );
   console.log( "BEGIN HOUSEHOLDER");
   // NEED TYPE CHECKING
   let [P,A,QT] = householder_reduction ( tf.tensor2d(M).arraySync() );
   P .print()
   A .print()
   QT.print()
   tf.dot( tf.dot(P,A),QT ).print()
   console.log( 'THIS THING NOW. NUMERICAL ACCURACY IS UNDERWHELMING' )
   //console.log(A)
   diagonalize_tridiagonal( A.arraySync() )// A.arraySync() )
   //VT.print()
   //console.log( "K(=0)TH HOUSEHOLDER REDUCTION");
   //kth_householder( tf.tensor2d(A) , 2 );
   console.log( "END HOUSEHOLDER");
   rt_ = tf.tensor2d( [  [3,1] , [5,10] ] );
   console.log(rt_.arraySync()[0][0])
   console.log(rt_.arraySync()[1][0])
   let a = index_value(rt_,0,0);
   let b = index_value(rt_,1,0);
   rich_rot( a , b ).print()
   rich_rot( tf.scalar(0) , b ).print()
   diagonalize_2b2( rt_ ).print()
}

run();
