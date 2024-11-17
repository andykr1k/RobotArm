#include "Wire.h"
#include "Adafruit_PWMServoDriver.h"
#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);
#define MIN_PULSE_WIDTH 1000
#define MAX_PULSE_WIDTH 2000
#define FREQUENCY 50
#define SMOOTH_STEP_DELAY 50
#define DEADZONE_THRESHOLD 2

const char *ssid = "Alight Riverside";
const char *password = "TerryDiscover";

int currentAngles[16];
int targetAngles[16];
unsigned long lastUpdateTime = 0;

ESP8266WebServer server(80);

void processCommands(String commands);
void processCommand(String command);
void updateServoPositions();

void setup()
{
  Serial.begin(9600);
  delay(1000);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid);
  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected to Wi-Fi");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  pwm.begin();
  pwm.setPWMFreq(FREQUENCY);

  for (int i = 0; i < 16; i++)
  {
    currentAngles[i] = 90;
    targetAngles[i] = 90;
  }

  server.on("/", []()
            { server.send(200, "text/plain", "ESP8266 Servo Controller"); });

  server.on("/control", []()
            {
    if (server.hasArg("commands")) {
      String commands = server.arg("commands");
      processCommands(commands);
      server.send(200, "text/plain", "Commands processed");
    } else {
      server.send(400, "text/plain", "Missing 'commands' parameter");
    } });

  server.begin();
  Serial.println("HTTP server started");
}

int pulseWidth(int angle)
{
  int pulse_width = map(angle, 0, 180, 143, 589);
  return pulse_width;
}

void processCommands(String commands)
{
  commands.trim();
  if (commands.startsWith("{") && commands.endsWith("}"))
  {
    commands = commands.substring(1, commands.length() - 1);
    commands.replace(" ", "");
    int commaIndex;

    while ((commaIndex = commands.indexOf(',')) != -1)
    {
      String command = commands.substring(0, commaIndex);
      processCommand(command);
      commands = commands.substring(commaIndex + 1);
    }

    processCommand(commands);
  }
  else
  {
    Serial.println("Invalid command format. Use {4:90,5:50}");
  }
}

void processCommand(String command)
{
  int colonIndex = command.indexOf(':');
  if (colonIndex != -1)
  {
    int motorID = command.substring(0, colonIndex).toInt();
    int angle = command.substring(colonIndex + 1).toInt();

    if (motorID >= 0 && motorID <= 15 && angle >= 0 && angle <= 180)
    {
      Serial.println("Moving Servo: " + String(motorID) + " to " + String(angle));
      targetAngles[motorID] = angle;
    }
    else
    {
      Serial.println("Invalid command. Use format MOTOR_ID:ANGLE (e.g., 4:90)");
    }
  }
  else
  {
    Serial.println("Invalid command format. Use MOTOR_ID:ANGLE");
  }
}

void updateServoPositions()
{
  unsigned long currentTime = millis();

  if (currentTime - lastUpdateTime >= SMOOTH_STEP_DELAY)
  {
    for (int i = 0; i < 16; i++)
    {
      int current = currentAngles[i];
      int target = targetAngles[i];

      if (abs(current - target) > DEADZONE_THRESHOLD)
      {
        if (current < target)
        {
          currentAngles[i]++;
        }
        else if (current > target)
        {
          currentAngles[i]--;
        }

        int pwm_value = pulseWidth(currentAngles[i]);
        pwm.setPWM(i, 0, pwm_value);
      }
    }
    lastUpdateTime = currentTime;
  }
}

void loop()
{
  server.handleClient();
  updateServoPositions();
}
