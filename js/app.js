 function init()
{
  var lselect = d3.select("#select-league");
  
  d3.json("/lnames").then((lgnames) => {
    
    lgnames.forEach((lname) => {
      
      lselect
        .append("option")
        .text(lname)
        .property("value", lname);
    });

  
    newleague = d3.select("#select-league").property("value")    
    var tselect = d3.select("#select-season");
    tselect.html("")
    var url = "/snames/" + newleague;

    d3.json(url).then((snnames) => {
      
      snnames.forEach((snname) => {
        
        tselect
          .append("option")
          .text(snname)
          .property("value", snname);
      });    
    });
    
    newseason = '2008-2009'
    lgdetails(newleague)
    countrymap()
    tschart(newleague, newseason)
    pchart(newleague, newseason)
    bchart(newleague, newseason)

  });  
};

// function submitbutton()
// {  
//   var tselect = d3.select("#result");
//   tselect.html("")

//   pwage = d3.select("#pwage").property("value")   
//   empduration = d3.select("#empduration").property("value")
//   fpt = d3.select("#ftposition").property("value")


//   var url = "/predict" + userinput;

//   d3.json(url).then((resultset) => {    
//     snnames.forEach((resultdata) => {      
//       tselect
//         .append("option")
//         .text(snname)
//         .property("value", snname);
//     });    
//   });    
// }

// function leagueChanged(newleague) {  
//   newleague = d3.select("#select-league").property("value")   
//   newseason = d3.select("#select-season").property("value")
//   snames(newleague)
//   lgdetails(newleague)    
//   tschart(newleague, newseason)
//   pchart(newleague, newseason) 
//   bchart(newleague, newseason) 
// }

// function seasonChanged(newSeason) {
//   newleague = d3.select("#select-league").property("value")   
//   newseason = d3.select("#select-season").property("value")       
//   tschart(newleague, newseason)
//   pchart(newleague, newseason)
//   bchart(newleague, newseason)
// }

// function countrymap()
// {
      
//     var myMap = L.map("country-map", { 
//       center: [48, 4],
//       zoom: 4
//     });

//     L.tileLayer("https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token={accessToken}", {
//       attribution: "Map data &copy; <a href=\"https://www.openstreetmap.org/\">OpenStreetMap</a> contributors, <a href=\"https://creativecommons.org/licenses/by-sa/2.0/\">CC-BY-SA</a>, Imagery © <a href=\"https://www.mapbox.com/\">Mapbox</a>",
//       container:'country-map',
//       maxZoom: 18,
//       id: "mapbox.streets-basic",
//       accessToken: API_KEY
//     }).addTo(myMap);
    
    

//     d3.json("/clist").then((clist) => {  
      
//         var countries = []
        
//         clist.forEach(function(d) {
//           countries.push({
//             "country": d.country,
//             "location": d.location,
//             "count":d.count
//           })
//         })

      
//         // Loop through the cities array and create one marker for each city object
//         for (var i = 0; i < countries.length; i++) {

//           // Conditionals for countries points
//           var color = "";
//           if (countries[i].count > 100) {
//             color = "yellow";
//           }
//           else if (countries[i].count > 200) {
//             color = "blue";
//           }
//           else if (countries[i].count > 300) {
//             color = "green";
//           }
//           else {
//             color = "red";
//           }

      
//           L.circle(countries[i].location, {
//             fillOpacity: 0.75,
//             color: "white",
//             fillColor: color,
//             // Adjust radius
//             // radius: parseint(countries[i].matches) * 1500
//             radius: 50  * 1500
//           }).bindPopup("<h4 text-align: center>" + countries[i].country + "</h4> Number of matches: " + countries[i].count).addTo(myMap);
//         }
          
//       });  

// }


// function lgdetails(newleague)
// {   
//   var imgselect = d3.select("#leaguelogo");
//   var lgtext = d3.select("#lgtext");

//   var url = "/lgdetails/" + newleague;

//   d3.json(url).then((lgtext) => {        
//         imgsrc= "/static/images/" + lgtext[9]         
//         imgselect.attr('src', imgsrc)        
//         d3.select("#lgtext")                  
//           .html(lgtext[10])
//   });

// }


// function tschart(newleague, newseason){

//   var url = "/tschart/" + newleague +'/'+ newseason;
  
//   d3.json(url).then((tsdata) => { 
//     var chdata = []
        
//         tsdata.forEach(function(d) {
//           chdata.push({
//             "date": d.date,
//             "count": d.id,            
//           })
//         })        
      
//       var data1 = {
//       labels: chdata.map(function(d) {
//         return d.date;
//       }),
//       series: [chdata.map(function(d) {
//         return d.count;
//       })]
//     };

//       Chart.defaults.global.defaultFontColor = 'black';
//       var chart = new Chartist.Line('.ct-chart', data1, 
//       {
//         fullWidth: true,
//         chartPadding: {
//           right: 40
//         }
//       }      
//       );
      
//     });

// }

// function pchart(newleague, newseason){

//   var url = "/pchart/" + newleague +'/'+ newseason;
//   // var svg = dimple.newSvg("#piechart", 90, 400);
//   piedata = []
//   d3.select("#pieChart").html("")
//     d3.json(url).then((pcdata) => { 
//       pcdata.forEach(function(d,i){

//         piedata.push({

//           label: d.team,
    
//           value: +d.matchcnt
//         })
//       })
      
//         var pie = new d3pie("pieChart", {                
//           "size": {    
//             "canvasHeight": 400,    
//             "canvasWidth": 350    
//           },
      
//           "data": {    
//             "content": piedata    
//           },
      
//           "labels": {    
//             "outer": {    
//               "pieDistance": 2   
//             }    
//           },    
//           "tooltips": {
//             "enabled": true,
//             "type": "placeholder",
//             "string": "{value} Matches",
//             "styles": {
//                 "backgroundColor": "#040404",
//                 "borderRadius": 5
//             }
//         }
//       });  
//     });    
  
// }


// function bchart(newleague, newseason){

//   var url = "/bchart/" + newleague +'/'+ newseason;
  
//   bardata = []
//   d3.select("#barchart").html("")
  
//   var svg = dimple.newSvg("#barchart", 450, 400);

//   d3.json(url).then((bcdata) => { 
//     bcdata.forEach(function(d,i){
//       bardata.push({
//         label: d.home_Team,  
//         value: +d.home_team_goal
//       })
//     })

    
        
//     var myChart = new dimple.chart(svg, bardata);
//     xAxis = myChart.addCategoryAxis("y", "label"),
//     yAxis = myChart.addMeasureAxis("x", "value"),    
//     yAxis.addOrderRule("value");
//     myChart.addSeries("", dimple.plot.bar);
//     myChart.setBounds(50, 10, 350, 330)
//     xAxis.hidden = false;
//     yAxis.hidden = false;
//     yAxis.fontSize = 10;
//     xAxis.fontSize = 8;
//     myChart._widthPixels(300)    
//     myChart.draw();
  
    
//     });    
  
// }

// // Initialize the dashboard
// var newleague;
// var newseason;
init();
