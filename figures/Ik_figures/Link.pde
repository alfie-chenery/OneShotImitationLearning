class Link{
  PVector start;
  PVector end;
  
  
  Link(PVector a, PVector b){
    start = PVector.mult(a, gridSize);
    end = PVector.mult(b, gridSize);
  }
  
  
  PVector getDir(){
    return PVector.sub(end, start).normalize(); 
  }
  
  void show(boolean showDir){
    stroke(255,128);
    strokeWeight(30);
    line(start.x, start.y, end.x, end.y);
    
    if(showDir){
      stroke(255,0,0);
      strokeWeight(2);
      int arrowLength = 75;
      arrow(end, PVector.add(end, PVector.mult(getDir(), arrowLength)));
      
    }
  }
  
  
  void arrow(PVector start, PVector end){
    pushMatrix();
    
    translate(start.x, start.y);
    
    float theta = PVector.angleBetween(PVector.sub(end,start), new PVector(1,0));
    float len = PVector.dist(start,end);
    float arrowSize = 5;
    
    //if arrow points upwards angle calculated is acute angle anticlockwise
    //rotate always treats angle as clockwise so theta needs inverting
    theta = (end.y < start.y ? -theta : theta);
      
    rotate(theta);
    
    line(0,0,len,0);
    noFill();
    beginShape();
    vertex(len-(2 * arrowSize), -arrowSize);
    vertex(len,0);
    vertex(len-(2 * arrowSize), arrowSize);
    endShape();
    
    popMatrix(); 
  }
  
}
