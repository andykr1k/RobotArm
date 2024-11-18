#include "Wire.h"
#include "Adafruit_PWMServoDriver.h"
#include <math.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

#define MIN_PULSE_WIDTH 600
#define MAX_PULSE_WIDTH 2400
#define FREQUENCY 60
#define SMOOTH_STEP_DELAY 10 // Time in milliseconds between steps

float currentAngles[16];
float targetAngles[16];
unsigned long lastUpdateTime[16]; // Track the last update time for each servo

void setup()
{
  Serial.begin(9600);
  pwm.begin();
  pwm.setPWMFreq(FREQUENCY);
  delay(500);

  Serial.println("Serial communication started. Send commands in the format {4:90,5:50}");
  Serial.println("Example: {4:90,5:50}");

  for (int i = 0; i < 16; i++)
  {
    currentAngles[i] = 90;
    targetAngles[i] = 90;
    lastUpdateTime[i] = millis(); // Initialize last update times
  }
}

int pulseWidth(float angle)
{
  return map(angle, 0, 180, 147, 589);
}

void processCommands(String commands)
{
  commands.trim();
  if (commands.startsWith("{") && commands.endsWith("}"))
  {
    commands = commands.substring(1, commands.length() - 1);
    int commaIndex = commands.indexOf(',');

    while (commaIndex != -1)
    {
      String command = commands.substring(0, commaIndex);
      commands = commands.substring(commaIndex + 1);
      processCommand(command);
      commaIndex = commands.indexOf(',');
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
    float angle = command.substring(colonIndex + 1).toFloat();

    if ((motorID >= 0 && motorID <= 15 && angle >= 0 && angle <= 180))
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

  for (int i = 0; i < 16; i++)
  {
    if (currentAngles[i] != targetAngles[i])
    {
      // Check if it's time to update this servo
      if (currentTime - lastUpdateTime[i] >= SMOOTH_STEP_DELAY)
      {
        if (currentAngles[i] < targetAngles[i])
        {
          currentAngles[i] += 1; // Increment by small steps
        }
        else if (currentAngles[i] > targetAngles[i])
        {
          currentAngles[i] -= 1; // Decrement by small steps
        }

        int pwm_value = pulseWidth(currentAngles[i]);
        pwm.setPWM(i, 0, pwm_value);

        // Update the last update time for this servo
        lastUpdateTime[i] = currentTime;
      }
    }
  }
}

void loop()
{
  // Check for incoming commands
  if (Serial.available())
  {
    String command = Serial.readStringUntil('\n');
    processCommands(command);
  }

  // Update servo positions continuously
  updateServoPositions();
}
