
async function plotbar() {

const data = [
   { index: 'beer', value: 50 },
   { index: 1, value: 100 },
   { index: 'cheese', value: 150 },
  ];

// Render to visor
const surface = { name: 'Bar chart', tab: 'Charts' };
names = ['bar1','bar2','bar3']
tfvis.render.barchart( surface, data , {color:['#ff0000','#0f0f00','#0000ff']} );

const series = ['First', 'Mob','Zombies'];

const serie1 = [] ;
const serie2 = [] ;
const serie3 = [] ;

for (let i = 0; i < 100; i++) {
  serie1[i] = {x:i, y:Math.random() * 100} ;
  serie2[i] = {x:i, y:Math.random() * 100} ;
  serie3[i] = {x:i, y:Math.random() * 100} ;
}

const scat_data = {values: [serie1, serie2, serie3 ], series:series}
scat_surface = {name:'Scatter',tab:'Graphs'}
tfvis.render.scatterplot(scat_surface, scat_data , { seriesColors: ['#ff0000','#0000ff','#f00ff0'] } );

const h_data = {
   values: [[4, 2, 8, 20], [1, 7, 2, 10], [3, 3, 20, 13]],
   xTickLabels: ['cheese', 'pig', 'font'],
   yTickLabels: ['speed', 'smoothness', 'dexterity', 'mana'],
}

// Render to visor
const h_surface = { name: 'Heatmap w Custom Labels', tab: 'Charts' };
tfvis.render.heatmap( h_surface, h_data, { colorMap:'viridis' } );
// greyscale , blues, viridis

const headers = [
  'Col 1',
  'Col Z',
  'Col 3',
];

const t_values = [
  [1, 2, 3],
  ['4', '5', '6'],
  ['<strong>7</strong>', true, false],
];


const t_surface = { name: 'Table', tab: 'Tables' };
tfvis.render.table( t_surface, { headers, values:t_values } );

const l_surface = { name: 'Lines' ,tab:'Graphs' } ;
let l_values = [[
  {x: 1, y: 20},
  {x: 2, y: 30},
  {x: 3, y: 15},
  {x: 4, y: 12}],
[
  {x: 1, y: 10},
  {x: 2, y: 5},
  {x: 3, y: -5},
  {x: 4, y: 20}],
];
tfvis.render.linechart( l_surface, { series:['s1','s2'] , values:l_values} ,
	 {xLabel:"xXx",yLabel:'yYy' , seriesColors:['#FF0000','#00ff00']});

}

function determineMeanAndStddev(data) {
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
//

function index_value(C,i,j) {
  var CS	= C.slice([i,j],[1,1]);
  return ( tf.max(CS) );
}

function kth_householder( A , k ) {
  var B		= A ;
  var k0	= k ;
  var k1	= k+1 ;
  var Op5	= tf.scalar(0.5);
  var s_	= index_value( B , k1 , k0 ) ;
  var alpha	= tf.scalar( 2 * ( +  s_ < 0 ) - 1 ) ;
  alpha = alpha.mul( tf.sqrt( tf.sum(B.transpose().slice([k0,k1],[1,-1]).square()) ) ) ;
  var r = tf.sqrt( Op5.mul( alpha.square().sub( alpha.mul(index_value(B,k1,k0)) ) ) );
  //
  //		THIS COMING LINE NEEDS RETHINKING <<SYNC>>
  var d = tf.reshape( tf.max( B.shape[0] ).sub(k1+1) , [1] ).dataSync();
  var v_ = tf.zeros([k1]) ;
  v_ = tf.concat( [v_ , tf.reshape( index_value(B,k1,k0).sub(alpha).mul(Op5).div(r) , [1] ) ], axis=0 );
  v_ = tf.concat( [v_,tf.reshape( Op5.div(r).mul(B.slice([k1+1,k0],[-1,1])), d ) ] , axis=0 )
  var Pk = tf.eye( B.shape[0] ).sub( tf.scalar(2).mul( tf.outerProduct(v_,v_.transpose()) ) )
  var Qk = Pk
  if ( B.shape[0] != B.shape[1] ) {
     alpha	= tf.scalar( 2 * ( + index_value(B,k0,k1) < 0 ) - 1 ) ;
     alpha	= alpha.mul( tf.sqrt( tf.sum(B.slice([k0,k1],[1,-1]).square()) ) );
     r = tf.sqrt( Op5.mul( alpha.square().sub( alpha.mul(index_value(B,k0,k1)) ) ) );
     //            THIS COMING LINE NEEDS RETHINKING <<SYNC>>
     var p  = tf.reshape( tf.max( B.shape[1] ).sub(k1+1) , [1] ).dataSync();
     var w_ = tf.zeros([k1]) ;
     w_ = tf.concat( [w_ , tf.reshape( index_value(B,k0,k1).sub(alpha).mul(Op5).div(r) , [1] ) ] , axis=0 );
     w_ = tf.concat( [w_ , tf.reshape( Op5.div(r).mul(B.slice([k0,k1+1],[1,-1]))    , p   ) ] )
     Qk = tf.eye( B.shape[1] ).sub( tf.scalar(2).mul( tf.outerProduct(w_,w_.transpose()) ) )
  }
  var Ak = tf.dot( tf.dot( Pk,B ),Qk );
  //
  // NOT WITH ASYNC
  return [Pk,Ak,Qk] ;
}

// export
// async
function householder_reduction ( M ) {
    var A = tf.tensor2d(M) ;
    var nlim =  tf.reshape( tf.max( A.shape[1] ) .sub(1) , [1] ).dataSync();
    if ( A.shape[0] < 2 )  {
        return ( A ) ;
    }
    let [P0,A0,Q0] = kth_householder( A , k=0 );
    if ( A.shape[0] == 2 ) {
        return ( [P0,A0,Q0] );
    }
    for (var k=1 ; k<nlim ; k++ ) {
        let [P1, A1, Q1] = kth_householder( A0 , k=k )
        A0 = A1
        P0 = tf.dot( P0 , P1 )
        Q0 = tf.dot( Q0 , Q1 )
    }
    var P = P0 ;
    var A = A0 ;
    var QT= Q0.transpose() ;
    return ( [P,A,QT] );
}

function listToMatrix( a ) {
     A = a.map( (_,i) =>
         a.map( (v,j) =>
            i==j ? v: 0 ));
   return ( A );
}

function matrixToVector(M,ishift=0) {
    // TYPE CHECKING NEEDED
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

function diagonalize_tridiagonal( T ) {
   T = tf.tensor2d(T);
   T .print();
   ci = matrixToVector( T.arraySync() , ishift =-1 )
   ai = matrixToVector( T.arraySync() , ishift = 0 )
   bi = matrixToVector( T.arraySync() , ishift = 1 )
   console.log( tf.sqrt( bi.mul(ci) ).dataSync() )
   sym_subdiagonals = tf.sqrt( bi.mul(ci) ).mul( tf_sgn( bi ) )
   console.log( sym_subdiagonals.dataSync() )
}

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

   plotbar();
   //console.log( SVDJS.SVD( M ) );
   console.log( "BEGIN HOUSEHOLDER");
   // NEED TYPE CHECKING
   let [P,A,QT] = householder_reduction ( tf.tensor2d(M).arraySync() );
   //U.print()
   A.print()
   //console.log(A)
   diagonalize_tridiagonal( A.arraySync() )
   //VT.print()
   //console.log( "K(=0)TH HOUSEHOLDER REDUCTION");
   //kth_householder( tf.tensor2d(A) , 2 );
   console.log( "END HOUSEHOLDER");

   /*
  */
}

run();
