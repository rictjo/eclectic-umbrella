/*
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
*/

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

/*
<RR 3 5
(0.5144957554275269, 0.8574929257125441, 5.830951894845301)
>
*/
function rich_rot( a , b ) { // TRY TO BUILD A TIDY FUNCTION
    let R = tf.tidy( () => {
        var c = tf.scalar(0);
        var s = tf.scalar(0);
        var r = tf.scalar(0);
        if ( !( a.equal(tf.scalar(0)) & b.equal(tf.scalar(0)) ) ) {
            r = tf.sqrt( a.mul(a) .add( b.mul(b) ) ) ;
            if ( a .equal( tf.scalar(0) ).dataSync()[0] ) { // ARGH...
                s = r.div( b ) ;
            } else {
                s = b.div( r ) ;
                c = r.sub( s.mul(b) ) .div(a) ;
            }
        }
        let cds = c.dataSync()[0]               // ARGH...
        let nsd = s.mul(-1).dataSync()[0]       // ARGH...
        let sds = s.dataSync()[0]               // ARGH...
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

function skew_eye(shape) {
   let BT = tf.tidy( () => {
   // BUFFERS ARE MUTABLE. TENSORS ARE NOT
   let nm = shape
   var m0 = nm[0] <= nm[1] ? nm[0] : nm[1]
   const buffer = tf.buffer(shape);
   for ( var i=0 ; i<m0 ; i++ ) {
      buffer.set( 1, i, i );
   }
   return ( buffer.toTensor() );
   });
   return ( BT );
}

function set_values( values , indices , shape ) {
   let BT = tf.tidy( () => {
   let nm = shape
   var kl = indices
   const buffer = tf.buffer(shape);
   for ( var i=0 ; i<indices.length ; i++ ) {
      buffer.set( values[i] , indices[i][0] , indices[i][1] );
   }
   return ( buffer.toTensor() );
   });
   return ( BT );
}

function diagonalize_tridiagonal( tridiagonal , maxiter=1000 , TOL=1E-10 , maxi22=100 , TOL22=1E-8 ) {
   //
   // THIS IS A SLOW WAY OF DIAGONALIZING A TRIDIAGONAL MATRIX
   //
   let [G,S,HT] = tf.tidy( () => {

   let S = tf.tensor2d( tridiagonal ).clone()
   let nm = S.shape
   var m0 = nm[0] <= nm[1] ? nm[0] : nm[1] - 1
   sI = skew_eye ( [ nm[0] , nm[0] ] );
   tI = skew_eye ( [ nm[1] , nm[1] ] );
   zI = skew_eye ( nm );
   let GI = tf.clone(sI) ;
   let HI = tf.clone(tI) ;
   for ( var k=0 ; k<maxiter ; k++ ) {
      for ( var i=0  ; i<m0 ; i++ ) {
         sI_ = sI .clone() ;
         tI_ = tI .clone() ;
         A   = S.slice( [i,i] , [2,2] )
         let [G,Z,H] = diagonalize_2b2 ( A , TOL=TOL );
         sI_ = set_values( G.sub(tf.eye(2)).dataSync() , [[i,i],[i,i+1],[i+1,i],[i+1,i+1]] , sI_.shape  ).add(tf.eye( sI_.shape[0] ) )
         tI_ = set_values( H.sub(tf.eye(2)).dataSync() , [[i,i],[i,i+1],[i+1,i],[i+1,i+1]] , tI_.shape  ).add(tf.eye( tI_.shape[0] ) )
         GI  = tf.dot( sI_ , GI )
         HI  = tf.dot( tI_ , HI )
         S = tf.dot( tf.dot( sI_ , S ) , tI_.transpose() )
         var n = m0 + 1 - i ;
         ran = tf.range(2,n).dataSync(); // ARGH ...
         for ( var jr=0 ; jr<ran.length ; jr++ ) {
            BS = S.bufferSync() ;
            var ii      = i;
            var jj      = i + ran[jr];
            var idx     = [ [ii,ii] , [ii,jj] , [jj,ii] , [jj,jj] ];
            var jdx     = [ (0,0),(0,1),(1,0),(1,1) ];
            A_ = tf.tensor2d( [ BS.get(ii,ii) , BS.get(ii,jj) , BS.get(jj,ii), BS.get(jj,jj)] , [2,2] )
            let [G,Z,H] = diagonalize_2b2 ( A_ , maxiter=maxi22, TOL=TOL22 );
            sI_ = set_values( G.sub(tf.eye(2)).dataSync() , idx , sI_.shape  ).add(tf.eye( sI_.shape[0] ) ); // ARGH ...
            tI_ = set_values( H.sub(tf.eye(2)).dataSync() , idx , tI_.shape  ).add(tf.eye( tI_.shape[0] ) ); // ARGH ...
            GI = tf.dot( sI_ , GI ); // FIRST PASS OK
            HI = tf.dot( tI_ , HI ); // FIRST PASS OK
            S  = tf.dot( tf.dot( sI_ , S ) , tI_.transpose() );
         };
      }
      var error = tf.sum(  matrixToVector( S.arraySync(), ishift=1 ).square().add( matrixToVector( S.arraySync(), ishift=-1 ).square() )  )
      if ( error < TOL ) {
          break;
      }
   }
   return [ GI.transpose(),S,HI ] ;
   });
   return [  G,S,HT ] ;
}

function nativeSVD( M ) {
    // NOTE THAT THIS IS SLOW
    let [ U , S , VT ] = tf.tidy( () => { // TIDIER ?
    let [ P , A , QT ] = householder_reduction ( tf.tensor2d(M).arraySync() );
    let [ G , S , HT ] = diagonalize_tridiagonal( A.arraySync() )
    return [tf.dot(P,G),S,tf.dot(QT,HT)];
  });
  return [U,S,VT];
}

function nativePCA( data_tf , axis=0 ) {
   res = determineMeanAndStddev( data_tf ,  axis=axis )
   std_dat = standardizeTensor( data_tf, res['dataMean'], res['dataStd'] )
   let [feature_coordinates,singular_values,components] = nativeSVD( std_dat.arraySync() )
   components = components.transpose()
   return { feature_coordinates , singular_values , components };
}

function diagonalize_2b2( B , TOL = 1E-7 , maxiter=100 , bVerbose=false ) {
    let [G_,M0,H_] = tf.tidy( () => { // TIDIER ?
  // THE ACTUAL FUNCTION
  let M         = B.slice([0,0],[2,2]);
  let M0        = M ;
  var error     = tf.scalar(1) ;
  const maxit_  = maxiter;
  let tolerance = tf.scalar(TOL) ;      // tf.scalar(1.0).div(tf.scalar(3.0)).square().print()
  let G_        = tf.eye(2);
  let H_        = tf.eye(2);
  for ( var k = 0 ; k<maxit_ ; k++ ) {
      // LEFT
      let G0    = rich_rot( index_value(M0,0,0) , index_value(M0,1,0) );
      M         = tf.dot(G0,M0) ;
      G_        = tf.dot(G0,G_) ;
      // RIGHT
      M         = M.transpose();
      let H0    = rich_rot( index_value(M,0,0) , index_value(M,1,0) );
      M         = tf.dot(H0,M ) ;
      H_        = tf.dot(H0,H_) ;
      // BACK
      M0        = M.transpose() ;
      error     = tf.sqrt( index_value(M0,1,0).square() .add( index_value(M0,0,1).square() ) );
      // THIS MUST BE DONE ...
         if ( error.less( tolerance ).dataSync()[0] )  { // ARGH...
            if (bVerbose) {
               error.print()
               tolerance.print()
               console.log('TOLCHECK',  error.less( tolerance ).dataSync() )
            }
            break;
         }
      //
  }
  return [G_,M0,H_];
  //RETURNED FROM TIDY
  });
  return [G_,M0,H_];
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
