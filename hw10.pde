// 參數設定
float amplitude = 150;    // 振幅 (A): 方塊偏離平衡點的最大距離
float period = 120;       // 週期 (T): 完成一次來回需要的 frame 數
float equilibriumX = 450; // 平衡點位置 (彈簧自然長度時的中心點)
float blockX;             // 方塊當前的 X 座標

void setup() {
  size(800, 400);
  rectMode(CENTER);       // 設定矩形以中心點定位
}

void draw() {
  background(200);        // 1. 背景色 (淺灰)
  
  // 2. 物理公式計算位置
  // 公式: x = 平衡點 + 振幅 * cos(omega * t)
  // omega = 2 * PI / period
  // t = frameCount
  float angle = TWO_PI * frameCount / period;
  blockX = equilibriumX + amplitude * cos(angle);
  
  // 3. 繪製牆壁 (左側固定端)
  stroke(255);
  strokeWeight(3);
  line(50, height/2 - 60, 50, height/2 + 60); // 垂直牆面
  
  // 4. 繪製彈簧 (動態鋸齒線)
  drawSpring(50, height/2, blockX - 30, height/2);
  
  // 5. 繪製方塊 (藍色)
  noStroke();
  fill(0, 0, 255);        // 純藍色
  rect(blockX, height/2, 60, 60);
}

// 輔助函式: 繪製彈簧
// (startX, startY): 彈簧左端固定點
// (endX, endY): 彈簧右端連接方塊的點
void drawSpring(float startX, float startY, float endX, float endY) {
  noFill();
  stroke(255);            // 白色線條
  strokeWeight(3);
  
  int segments = 40;      // 彈簧的節數 (鋸齒數量)
  float len = endX - startX;
  float segLen = len / segments;
  
  beginShape();
  vertex(startX, startY); // 起點
  
  // 繪製中間的鋸齒
  for (int i = 1; i < segments; i++) {
    float x = startX + i * segLen;
    float yOffset = 0;
    
    // 讓頭尾兩端稍微平緩一點，中間劇烈震盪
    if (i > 2 && i < segments - 2) {
      // 使用模數運算 (%) 來製造上下交替的鋸齒
      yOffset = (i % 2 == 0) ? 20 : -20;
    }
    
    vertex(x, startY + yOffset);
  }
  
  vertex(endX, endY);     // 終點
  endShape();
}
