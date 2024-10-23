#include "Wire.h"
#include "Adafruit_PWMServoDriver.h"
#include <math.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

#define MIN_PULSE_WIDTH 600
#define MAX_PULSE_WIDTH 2400
#define FREQUENCY 60
#define SMOOTH_STEP_DELAY 10

int currentAngles[16];
int targetAngles[16];

void setup()
{
  Serial.begin(9600);

  pwm.begin();
  pwm.setPWMFreq(FREQUENCY);

  delay(500);
  Serial.println("Serial communication started. Send commands in the format {4:90,5:90}");
  Serial.println("Example: {4:90,5:50}");

  currentAngles[0] = 90;
  targetAngles[0] = 90;

  for (int i = 1; i < 16; i++)
  {
    currentAngles[i] = 0;
    targetAngles[i] = 0;
  }
}

int pulseWidth(int angle)
{
  int pulse_width = map(angle, 0, 180, 147, 589);
  return pulse_width;
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
      String command = commands.substring(0, commaIndex - 1);
      commands = commands.substring(commaIndex + 2);
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
    int angle = command.substring(colonIndex + 1).toInt();

    if ((motorID >= 1 && motorID <= 15 && angle >= 0 && angle <= 110) || (motorID == 0 && angle >= 0 && angle <= 180))
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
  for (int i = 0; i < 16; i++)
  {
    int current = currentAngles[i];
    int target = targetAngles[i];

    if (current != target)
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
      delay(SMOOTH_STEP_DELAY);
    }
  }
}

void loop()
{
  if (Serial.available())
  {
    String command = Serial.readStringUntil('\n');
    processCommands(command);
  }

  updateServoPositions();
}
