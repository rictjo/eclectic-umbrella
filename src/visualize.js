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
//
const moduleName = "visualize"

async function plotbar() {

const data = [
   { index: 'beer', value: 50 },
   //{ index: 'beer', value:150 },
   { index: 1, value: 100 },
   { index: 'cheese', value: 150 },
  ];

// Render to visor
const surface = { name: 'Bar chart', tab: 'Charts' };
var names = ['bar1','bar2','bar3']
tfvis.render.barchart( surface, data , {color:['#ff0000','#1ecd0f','#0f0f00','#0000ff']} );

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
var scat_surface = {name:'Scatter',tab:'Graphs'}
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

export { plotbar } ;

/*
async function run() {
  //
  // HAIRY PLOTTER
  //
  plotbar();
}

run();
*/
