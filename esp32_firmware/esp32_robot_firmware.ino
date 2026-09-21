/*
  =============================================================================
  AI Face Emotion-Controlled Robot Firmware (ESP32)
  =============================================================================
  Features:
  - Dual Wi-Fi Modes: Station (STA) with automatic fallback to Access Point (AP).
  - High-Speed UDP Datagram Receiver on port 4210 (<2ms latency).
  - Built-in Dead-Man Safety Watchdog (stops motors if packet lost for >350ms).
  - PWM Speed Control compatible with L298N, TB6612FNG, DRV8833, or L9110S.
  - Live Status LED Indicator.
  =============================================================================
*/

#include <WiFi.h>
#include <WiFiUdp.h>

// ============================================================================
// 1. WI-FI CONFIGURATION
// ============================================================================
// Set your router / mobile hotspot credentials here for Station (STA) mode:
const char* STA_SSID     = "YOUR_WIFI_SSID";
const char* STA_PASSWORD = "YOUR_WIFI_PASSWORD";

// Fallback Access Point (AP) mode credentials:
const char* AP_SSID      = "Robot-Emotion-AP";
const char* AP_PASSWORD  = "12345678";  // Must be >= 8 characters

// UDP Port (must match laptop configuration)
const unsigned int UDP_PORT = 4210;
WiFiUDP udp;

// ============================================================================
// 2. MOTOR DRIVER PIN DEFINITIONS (Default: L298N / TB6612FNG)
// ============================================================================
// Left Motor (Motor A)
const int PIN_ENA = 14;  // Speed PWM
const int PIN_IN1 = 27;  // Direction 1
const int PIN_IN2 = 26;  // Direction 2

// Right Motor (Motor B)
const int PIN_ENB = 12;  // Speed PWM
const int PIN_IN3 = 25;  // Direction 1
const int PIN_IN4 = 33;  // Direction 2

// Built-in status LED (GPIO 2 on most ESP32 Dev modules)
const int PIN_LED = 2;

// PWM Configuration for ESP32 LEDC
const int PWM_FREQ        = 1000;  // 1 kHz
const int PWM_RESOLUTION  = 8;     // 8-bit (0 - 255)
const int PWM_CHAN_A      = 0;
const int PWM_CHAN_B      = 1;

// ============================================================================
// 3. SAFETY WATCHDOG
// ============================================================================
// If no valid UDP packet is received within this duration, the bot halts!
const unsigned long WATCHDOG_TIMEOUT_MS = 350;
unsigned long lastPacketTimestamp = 0;
bool isMoving = false;

// Buffer for incoming UDP packets
char packetBuffer[128];

// ============================================================================
// MOTOR CONTROL PRIMITIVES
// ============================================================================
void setSpeed(int speedA, int speedB) {
  speedA = constrain(speedA, 0, 255);
  speedB = constrain(speedB, 0, 255);
  
  #if ESP_ARDUINO_VERSION >= ESP_ARDUINO_VERSION_VAL(3, 0, 0)
    ledcWrite(PIN_ENA, speedA);
    ledcWrite(PIN_ENB, speedB);
  #else
    ledcWrite(PWM_CHAN_A, speedA);
    ledcWrite(PWM_CHAN_B, speedB);
  #endif
}

void moveForward(int speed) {
  digitalWrite(PIN_IN1, HIGH);
  digitalWrite(PIN_IN2, LOW);
  digitalWrite(PIN_IN3, HIGH);
  digitalWrite(PIN_IN4, LOW);
  setSpeed(speed, speed);
  isMoving = true;
}

void moveBackward(int speed) {
  digitalWrite(PIN_IN1, LOW);
  digitalWrite(PIN_IN2, HIGH);
  digitalWrite(PIN_IN3, LOW);
  digitalWrite(PIN_IN4, HIGH);
  setSpeed(speed, speed);
  isMoving = true;
}

// Maximum turn speed limit for controlled, precise rotation (prevents aggressive spin-out)
const int MAX_TURN_SPEED = 145;

void turnLeft(int speed) {
  // Skid steer: left wheels backward, right wheels forward
  // Constrain turn speed so turns are smooth, slow, and precise
  int turnSpeed = constrain(speed, 0, MAX_TURN_SPEED);
  digitalWrite(PIN_IN1, LOW);
  digitalWrite(PIN_IN2, HIGH);
  digitalWrite(PIN_IN3, HIGH);
  digitalWrite(PIN_IN4, LOW);
  setSpeed(turnSpeed, turnSpeed);
  isMoving = true;
}

void turnRight(int speed) {
  // Skid steer: left wheels forward, right wheels backward
  // Constrain turn speed so turns are smooth, slow, and precise
  int turnSpeed = constrain(speed, 0, MAX_TURN_SPEED);
  digitalWrite(PIN_IN1, HIGH);
  digitalWrite(PIN_IN2, LOW);
  digitalWrite(PIN_IN3, LOW);
  digitalWrite(PIN_IN4, HIGH);
  setSpeed(turnSpeed, turnSpeed);
  isMoving = true;
}

void stopMotors() {
  digitalWrite(PIN_IN1, LOW);
  digitalWrite(PIN_IN2, LOW);
  digitalWrite(PIN_IN3, LOW);
  digitalWrite(PIN_IN4, LOW);
  setSpeed(0, 0);
  isMoving = false;
}

// ============================================================================
// SETUP
// ============================================================================
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n\n========================================");
  Serial.println("  AI Face Emotion-Controlled Bot ESP32  ");
  Serial.println("========================================");

  // Configure Motor Pins
  pinMode(PIN_IN1, OUTPUT);
  pinMode(PIN_IN2, OUTPUT);
  pinMode(PIN_IN3, OUTPUT);
  pinMode(PIN_IN4, OUTPUT);
  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, LOW);
  stopMotors();

  // Configure PWM
  #if ESP_ARDUINO_VERSION >= ESP_ARDUINO_VERSION_VAL(3, 0, 0)
    ledcAttach(PIN_ENA, PWM_FREQ, PWM_RESOLUTION);
    ledcAttach(PIN_ENB, PWM_FREQ, PWM_RESOLUTION);
  #else
    ledcSetup(PWM_CHAN_A, PWM_FREQ, PWM_RESOLUTION);
    ledcAttachPin(PIN_ENA, PWM_CHAN_A);
    ledcSetup(PWM_CHAN_B, PWM_FREQ, PWM_RESOLUTION);
    ledcAttachPin(PIN_ENB, PWM_CHAN_B);
  #endif

  // Attempt connection in STA mode
  Serial.print("Connecting to Wi-Fi: ");
  Serial.println(STA_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(STA_SSID, STA_PASSWORD);

  unsigned long startAttemptTime = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - startAttemptTime < 8000) {
    delay(250);
    digitalWrite(PIN_LED, !digitalRead(PIN_LED));
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[OK] Connected to Wi-Fi!");
    Serial.print("IP Address (Enter this in Laptop UI): ");
    Serial.println(WiFi.localIP());
    digitalWrite(PIN_LED, HIGH);
  } else {
    // Fallback to Access Point (AP) Mode
    Serial.println("\n[!] STA Connection timed out. Starting AP Hotspot mode...");
    WiFi.mode(WIFI_AP);
    WiFi.softAP(AP_SSID, AP_PASSWORD);
    Serial.print("Created Wi-Fi Hotspot: ");
    Serial.println(AP_SSID);
    Serial.print("AP IP Address: ");
    Serial.println(WiFi.softAPIP());
    digitalWrite(PIN_LED, HIGH);
  }

  // Start listening on UDP port
  udp.begin(UDP_PORT);
  Serial.print("Listening for UDP commands on port: ");
  Serial.println(UDP_PORT);
  Serial.println("System Ready!\n");
}

// ============================================================================
// MAIN LOOP
// ============================================================================
void loop() {
  // 1. Check for incoming UDP packet
  int packetSize = udp.parsePacket();
  if (packetSize > 0) {
    int len = udp.read(packetBuffer, sizeof(packetBuffer) - 1);
    if (len > 0) {
      packetBuffer[len] = '\0';
      lastPacketTimestamp = millis();  // Reset watchdog

      // Parse packet format: "CMD:SPEED:SEQ" (e.g., "F:200:1042")
      char cmd = 'S';
      int speed = 200;
      long seq = 0;

      char* token = strtok(packetBuffer, ":");
      if (token != NULL) {
        cmd = token[0];
        token = strtok(NULL, ":");
        if (token != NULL) {
          speed = atoi(token);
          token = strtok(NULL, ":");
          if (token != NULL) {
            seq = atol(token);
          }
        }
      }

      // Execute motion command
      switch (cmd) {
        case 'F':
          moveForward(speed);
          break;
        case 'B':
          moveBackward(speed);
          break;
        case 'L':
          turnLeft(speed);
          break;
        case 'R':
          turnRight(speed);
          break;
        case 'S':
        default:
          stopMotors();
          break;
      }
    }
  }

  // 2. Dead-Man Safety Watchdog:
  // If no packet has arrived within WATCHDOG_TIMEOUT_MS, halt robot immediately!
  if (isMoving && (millis() - lastPacketTimestamp > WATCHDOG_TIMEOUT_MS)) {
    Serial.println("[WATCHDOG TIMEOUT] No packets received! Halting motors for safety.");
    stopMotors();
  }
}
