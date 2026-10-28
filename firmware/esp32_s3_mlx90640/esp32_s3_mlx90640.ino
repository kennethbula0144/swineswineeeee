#include <Adafruit_MLX90640.h>
#include <HTTPClient.h>
#include <WiFi.h>
#include <Wire.h>

// Configure these values for the local Wi-Fi network and your ESP32-S3 board.
const char *WIFI_SSID = "YOUR_WIFI_SSID";
const char *WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char *BACKEND_URL = "http://YOUR_BACKEND_IP:8765/api/thermal/frame";
const char *THERMAL_TOKEN = "YOUR_LONG_RANDOM_THERMAL_TOKEN";

constexpr int I2C_SDA_PIN = 8;
constexpr int I2C_SCL_PIN = 9;
constexpr size_t PIXEL_COUNT = 32 * 24;
constexpr uint32_t POST_INTERVAL_MS = 1000;

Adafruit_MLX90640 thermalCamera;
float thermalPixels[PIXEL_COUNT];
uint32_t lastPostMs = 0;

void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to Wi-Fi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());
}

String makeFrameJson(const float *pixels, float ambientC) {
  String payload;
  payload.reserve(12000);
  payload = "{\"pixels\":[";

  for (size_t i = 0; i < PIXEL_COUNT; ++i) {
    if (i != 0) payload += ',';
    payload += String(pixels[i], 1);
  }

  payload += "],\"ambient_c\":";
  payload += String(ambientC, 1);
  payload += '}';
  return payload;
}

void postFrame(const float *pixels, float ambientC) {
  if (WiFi.status() != WL_CONNECTED) connectWiFi();

  WiFiClient client;
  HTTPClient http;
  http.setTimeout(5000);
  if (!http.begin(client, BACKEND_URL)) {
    Serial.println("Could not initialize the backend HTTP connection.");
    return;
  }

  http.addHeader("Content-Type", "application/json");
  http.addHeader("X-Thermal-Token", THERMAL_TOKEN);
  const String payload = makeFrameJson(pixels, ambientC);
  const int statusCode = http.POST(payload);
  if (statusCode > 0) {
    Serial.printf("Thermal frame POST: HTTP %d\n", statusCode);
  } else {
    Serial.printf("Thermal frame POST failed: %s\n", http.errorToString(statusCode).c_str());
  }
  http.end();
}

void setup() {
  Serial.begin(115200);
  delay(200);

  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  Wire.setClock(400000);

  if (!thermalCamera.begin(MLX90640_I2CADDR_DEFAULT, &Wire)) {
    Serial.println("MLX90640 not found at I2C address 0x33. Check SDA, SCL, power, and ground.");
    while (true) delay(1000);
  }

  thermalCamera.setMode(MLX90640_CHESS);
  thermalCamera.setResolution(MLX90640_ADC_18BIT);
  thermalCamera.setRefreshRate(MLX90640_2_HZ);

  connectWiFi();
  Serial.println("MLX90640-D110 thermal frame sender ready.");
}

void loop() {
  const int result = thermalCamera.getFrame(thermalPixels);
  if (result != 0) {
    Serial.printf("MLX90640 frame read failed: %d\n", result);
    delay(250);
    return;
  }

  const uint32_t now = millis();
  if (now - lastPostMs >= POST_INTERVAL_MS) {
    lastPostMs = now;
    postFrame(thermalPixels, thermalCamera.getTa());
  }
}
