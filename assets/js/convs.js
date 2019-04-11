function gridData(nrows, ncols) {
	var data = new Array();
	var xpos = 1; //starting xpos and ypos at 1 so the stroke will show when we make the grid below
	var ypos = 1;
	var width = 30;
	var height = 30;
	var click = 0;
	
	// iterate for rows	
	for (var row = 0; row < nrows; row++) {
		data.push( new Array() );
		
		// iterate for cells/columns inside rows
		for (var column = 0; column < ncols; column++) {
			data[row].push({
				x: xpos,
				y: ypos,
                                c: column,
                                r: row,
                                xc: xpos+width/2,
                                yc: ypos+height/2,
				width: width,
				height: height,
				click: click
			})
			// increment the x position. I.e. move it over by 50 (width variable)
			xpos += width;
		}
		// reset the x position after a row is complete
		xpos = 1;
		// increment the y position for the next row. Move it down 50 (height variable)
		ypos += height;	
	}
	return data;
}

function gridKernel(nrows, ncols) {
	var data = new Array();
	var xpos = 1; //starting xpos and ypos at 1 so the stroke will show when we make the grid below
	var ypos = 1;
	var width = 30;
	var height = 30;
	var click = 0;
	
        var values = [[0, 1, 0], [1, 0, 1], [0, 1, 0]];
	// iterate for rows	
	for (var row = 0; row < nrows; row++) {
		data.push( new Array() );
		
		// iterate for cells/columns inside rows
		for (var column = 0; column < ncols; column++) {
			data[row].push({
				x: xpos,
				y: ypos,
                                v: values[row][column],
                                c: column,
                                r: row,
                                xc: xpos+width/2,
                                yc: ypos+height/2,
				width: width,
				height: height,
				click: click
			})
			// increment the x position. I.e. move it over by 50 (width variable)
			xpos += width;
		}
		// reset the x position after a row is complete
		xpos = 1;
		// increment the y position for the next row. Move it down 50 (height variable)
		ypos += height;	
	}
	return data;
}

function gridDataKern(data, kern) {
        var nrows = 10;
        var ncols = 10;
	var data = new Array();
	var xpos = 1; //starting xpos and ypos at 1 so the stroke will show when we make the grid below
	var ypos = 1;
	var width = 30;
	var height = 30;
	var click = 0;
	
        var values = [[0, 1, 0], [1, 0, 1], [0, 1, 0]];
	// iterate for rows	
	for (var row = 0; row < nrows; row++) {
		data.push( new Array() );
		
		// iterate for cells/columns inside rows
		for (var column = 0; column < ncols; column++) {

                        value = 0;
                        for (var krow = 0; krow < 3; krow++) {
                            for (var kcol = 0; kcol < 3; kcol++) {
                                value += (column + row*nrows) * values[krow][kcol];
                            }
                        }
			data[row].push({
				x: xpos,
				y: ypos,
                                v: value,
                                c: column,
                                r: row,
                                xc: xpos+width/2,
                                yc: ypos+height/2,
				width: width,
				height: height,
				click: click
			})
			// increment the x position. I.e. move it over by 50 (width variable)
			xpos += width;
		}
		// reset the x position after a row is complete
		xpos = 1;
		// increment the y position for the next row. Move it down 50 (height variable)
		ypos += height;	
	}
	return data;
}

var data = gridData(10, 10);	

var grid = d3.select("#grid")
	.append("svg")
	.attr("width","310px")
	.attr("height","310px");
	
var row = grid.selectAll(".row")
	.data(data)
	.enter().append("g")
	.attr("class", "row");
	
var cells = row.selectAll(".square")
	.data(function(d) { return d; })
        .enter()
        .append("g");

cells.append("rect")
    .attr("class","square")
    .attr("x", function(d) { return d.x; })
    .attr("y", function(d) { return d.y; })
    .attr("c", function(d) { return d.c; })
    .attr("r", function(d) { return d.r; })
    .attr("width", function(d) { return d.width; })
    .attr("height", function(d) { return d.height; })
    .style("fill", "#fff")
    .style("stroke", "#222")
    .on('mouseover', test);

cells.append("text")
    .attr("class","text")
    .attr("x", function(d) { return d.xc; })
    .attr("y", function(d) { return d.yc + 5; })
    .text(function(d) { return d.c + d.r * 10; })
    .style("text-anchor", "middle");

var kern = gridKernel(3, 3);	
var grid = d3.select("#grid")
	.append("svg")
	.attr("width","100px")
	.attr("height","200px");
	
var row = grid.selectAll(".row")
	.data(kern)
	.enter().append("g")
	.attr("class", "row");
	
var kernCells = row.selectAll(".square-kern")
	.data(function(d) { return d; })
	.enter()
    
kernCells.append("rect")
	.attr("class","square-kern")
	.attr("x", function(d) { return d.x; })
	.attr("y", function(d) { return d.y; })
	.attr("c", function(d) { return d.c; })
	.attr("r", function(d) { return d.r; })
	.attr("width", function(d) { return d.width; })
	.attr("height", function(d) { return d.height; })
	.style("fill", "#fff")
	.style("stroke", "#222");

kernCells.append("text")
    .attr("class","text")
    .attr("x", function(d) { return d.xc; })
    .attr("y", function(d) { return d.yc + 5; })
    .text(function(d) { return d.v; })
    .style("text-anchor", "middle");

var data = gridDataKern(data, kern);	
var grid = d3.select("#grid")
	.append("svg")
	.attr("width","310px")
	.attr("height","310px");
	
var row = grid.selectAll(".row")
	.data(data)
	.enter().append("g")
	.attr("class", "row");
	
var outputCells = row.selectAll(".square-output")
	.data(function(d) { return d; })
	.enter()
    
outputCells.append("rect")
	.attr("class","square-output")
	.attr("x", function(d) { return d.x; })
	.attr("y", function(d) { return d.y; })
	.attr("c", function(d) { return d.c; })
	.attr("r", function(d) { return d.r; })
	.attr("width", function(d) { return d.width; })
	.attr("height", function(d) { return d.height; })
	.style("fill", "#fff")
	.style("stroke", "#222");

outputCells.append("text")
    .attr("class","text-output")
    .attr("x", function(d) { return d.xc; })
    .attr("y", function(d) { return d.yc + 5; })
    .text(function(d) { return ""; })
    .style("text-anchor", "middle");

d3.selectAll(".square-output")
    .filter(function(d) {
        return (d.r == 0 || d.c == 0 || d.r == 9 || d.c == 9);
    })
    .style("fill", "#aaa");
function test(d){
    // Ignore boundaries
    if (d.c == 0 || d.r == 0 || d.r == 9 || d.c == 9)
        return;

    var x = d3.select(this);
    var c = parseInt(x.attr("c"));
    var r = parseInt(x.attr("r"));

    var f = d3.selectAll(".square")
        .style("fill", "#fff");

    var f = d3.selectAll(".square-output")
        .style("fill", "#fff");

    var f = d3.selectAll(".text-output")
        .text(function(d){ return ""; });

    var f = d3.selectAll(".square")
        .filter(function(d) {
            return (d.r > r-2 &&  d.r < r+2 && d.c > c-2 && d.c < c+2);
        })
        .style("fill", "#add8e6");

    x.style("fill", "#ff0000");

    var f = d3.selectAll(".square-output")
        .filter(function(d) {
            return (d.r == r && d.c == c);
        })
        .style("fill", "#f00");

    var f = d3.selectAll(".square-output")
        .filter(function(d) {
            return (d.r == 0 || d.c == 0 || d.r == 9 || d.c == 9);
        })
        .style("fill", "#aaa");

    var f = d3.selectAll(".text-output")
        .filter(function(d) {
            return (d.r == r && d.c == c);
        })
        .text(function(d){
            return d.v;
        });

}

