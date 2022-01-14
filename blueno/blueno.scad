$fn=28;

//helper
module rounded_cylinder(r,h,n) {
  rotate_extrude(convexity=1) {
    offset(r=n) offset(delta=-n) square([r,h]);
    square([n,h]);
  }
}


module head() {
    earThickness = rands(0.4, 0.6, 1)[0];
    earWidth = 0.7;
    earLength = 1;
    noseSize = 0.13;
    noseThickness = rands(0.5, 1.1, 1)[0];
    noseWidth = 0.7;
    noseLength = 0.7;
    lampSize = rands(0.5, 0.9, 1)[0];
    
    hull() {
        translate([0,0,0]) 
            sphere(0.2);
        translate([0.06,0,0])
            sphere(0.18);
    }
    
    // fixed
    earX = 0;
    earY = 0.18;
    earZ = 0.12 ;
    earXDeg = 35;
    // right ear
    translate([earX,earY,earZ])
    rotate([earXDeg,0,0])
    scale([earThickness,earWidth,earLength])
        sphere(0.13);
    // left ear
    translate([earX,-earY,earZ])
    rotate([-earXDeg,0,0])
    scale([earThickness,earWidth,earLength])
        sphere(0.13);
    
    // nose
    translate([0.21,0,0.02])
    rotate([0,0,0])
    scale([noseThickness,noseWidth,noseLength])
        sphere(0.13);
        
    //lamp 
    translate([0.2,0,0.15])
    rotate([0,-30,0])
    scale([lampSize,lampSize,lampSize])
    difference() { 
        sphere(r=0.25); 
        translate([0, 0, -2])
            cube([4, 4, 4], center=true);
    }
}

module body() {
    hull() {
        translate([0,0,0.2]) 
            sphere(0.2);
        translate([0,0,-0.3])
        sphere(0.3);
    }
}

module arms() {
    length = rands(0.5, 0.8, 1)[0];
    radius = rands(0.1, 0.2, 1)[0];
    xDeg = 0;
    yDeg = rands(100, 120, 1)[0];
    zDeg = rands(40, 70, 1)[0];
    x = 0;
    y = 0.12;
    z = 0.2;
    
    translate([x, y, z])
    rotate([xDeg, yDeg, zDeg])
        rounded_cylinder(radius, length, (radius/2)-0.01);
    
    translate([x, -y, z])
    rotate([xDeg, yDeg, 360-zDeg])
        rounded_cylinder(radius, length, (radius/2)-0.01);
}

module legs() {
    length = rands(0.5, 0.8, 1)[0];
    radius = rands(0.1, 0.2, 1)[0];
    xDeg = 0;
    yDeg = rands(90, 110, 1)[0];
    zDeg = rands(15, 25, 1)[0];
    x = 0;
    y = 0.15;
    z = -0.4;
    
    translate([x, y, z])
    rotate([xDeg, yDeg, zDeg])
        rounded_cylinder(radius, length, (radius/2)-0.01);
    
    translate([x, -y, z])
    rotate([xDeg, yDeg, 360-zDeg])
        rounded_cylinder(radius, length, (radius/2)-0.01);
}

module blueno() {
    headSize = rands(1.2,1.8,1)[0];
    
    translate([0,0,0.5])
    rotate([0,20,0])
    scale([headSize,headSize,headSize])
        head();
    body();
    arms();
    legs();
}
translate([-0.3,0,-0.15])
    blueno();
//translate([1,0,0])
//head();
//color(alpha=0.2)
//    sphere(1);

//single_rand = rands(0,1,1)[0];
//echo(single_rand);
