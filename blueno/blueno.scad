$fn=64;


module rounded_cylinder(r,h,n) {
  rotate_extrude(convexity=1) {
    offset(r=n) offset(delta=-n) square([r,h]);
    square([n,h]);
  }
}

//scale([1,1.3,1])
//rounded_cylinder(r=10,h=25,n=4.9);

module body() {
    hull() {
        translate([0,0,0.5]) 
            sphere(0.35);
        sphere(0.5);
    }
}

body();

//single_rand = rands(0,1,1)[0];
//echo(single_rand);
