int gridSize = 50;

void setup(){
  //size(551, 401); //figure 1
  size(451,451); //figure 2
  background(0);
  
  //draw grid
  strokeWeight(1);
  stroke(255,50);
  for(int x=0; x<=width; x+=gridSize){
    line(x,0,x,height);
  }
  for(int y=0; y<=height; y+=gridSize){
    line(0,y,width,y);
  }
  
  Link l1;
  Link l2;
  Link l3;
  Link l4;
  
  //figure 1a
  //l1 = new Link(new PVector(1,6), new PVector(4,2));
  //l2 = new Link(new PVector(4,2), new PVector(6,1));
  //l3 = new Link(new PVector(6,1), new PVector(8,2));
  //l4 = new Link(new PVector(8,2), new PVector(9,3));
  
  //figure 1b
  //l1 = new Link(new PVector(1,6), new PVector(4,2));
  //l2 = new Link(new PVector(4,2), new PVector(6,3));
  //l3 = new Link(new PVector(6,3), new PVector(8,2));
  //l4 = new Link(new PVector(8,2), new PVector(9,3));
  
  //figure 1c
  //l1 = new Link(new PVector(1,6), new PVector(5,3));
  //l2 = new Link(new PVector(5,3), new PVector(6,1));
  //l3 = new Link(new PVector(6,1), new PVector(8,2));
  //l4 = new Link(new PVector(8,2), new PVector(9,3));
  
  
  
  //figure 2a
  //l1 = new Link(new PVector(2,2), new PVector(2,7));
  //l2 = new Link(new PVector(2,7), new PVector(5,7));
  //l3 = new Link(new PVector(5,7), new PVector(5,2));
  //l4 = new Link(new PVector(5,2), new PVector(6,2));
  
  //figure 2b
  l1 = new Link(new PVector(2,2), new PVector(5,6));
  l2 = new Link(new PVector(5,6), new PVector(2,6));
  l3 = new Link(new PVector(2,6), new PVector(5,2));
  l4 = new Link(new PVector(5,2), new PVector(6,2));
  
  
  l1.show(false);
  l2.show(false);
  l3.show(false);
  l4.show(true);
  
  //figure 1
  //textSize(20);
  //text("x=0\ny=0", 0.5 * gridSize, 7 * gridSize);
  //text("x=8\ny=3\ntheta=-45°", 8 * gridSize, 4 * gridSize);
  
  //figure 2
  textSize(20);
  text("x=0\ny=0", 0.5 * gridSize, 1 * gridSize);
  text("x=4\ny=0\ntheta=0°", 6.5 * gridSize, 3 * gridSize);
  
  
  noStroke();
  fill(255);
  
  //figure 1
  //circle(1 * gridSize, 6 * gridSize, gridSize * 0.75);
  //fill(255, 0, 0);
  //circle(9 * gridSize, 3 * gridSize, gridSize * 0.6);
  
  //figure 2
  circle(2 * gridSize, 2 * gridSize, gridSize * 0.75);
  fill(255, 0, 0);
  circle(6 * gridSize, 2 * gridSize, gridSize * 0.6);
  
  save("fig.png");
  
}
