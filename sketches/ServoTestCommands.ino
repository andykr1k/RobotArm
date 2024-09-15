#include "Wire.h"
#include "Adafruit_PWMServoDriver.h"

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

#define MIN_PULSE_WIDTH 600
#define MAX_PULSE_WIDTH 2925
#define FREQUENCY 50

void setup() {
  Serial.begin(9600);

  pwm.begin();
  pwm.setPWMFreq(FREQUENCY);

  delay(500);
  Serial.println("Serial communication started. Send commands in the format {4:90,5:90}");
  Serial.println("Example: {4:90,5:50} (for motors 4 and 5 at 90 and 50 degrees, respectively)");
}

int pulseWidth(int angle) {
  int pulse_wide = map(angle, 0, 180, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
  int analog_value = int(float(pulse_wide) / 1000000 * FREQUENCY * 4096);
  Serial.println(String(analog_value));
  return analog_value;
}

void processCommands(String commands) {
  commands.trim();
  if (commands.startsWith("{") && commands.endsWith("}")) {
    commands = commands.substring(1, commands.length() - 1);
    int commaIndex = commands.indexOf(',');

    while (commaIndex != -1) {
      String command = commands.substring(0, commaIndex - 1);
      commands = commands.substring(commaIndex + 2);
      processCommand(command);
      commaIndex = commands.indexOf(',');
    }
    
    processCommand(commands);
  } else {
    Serial.println("Invalid command format. Use {4:90,5:50}");
  }
}

void processCommand(String command) {
  Serial.println("Processing Command: " + command);
  
  int colonIndex = command.indexOf(':');
  if (colonIndex != -1) {
    int motorID = command.substring(0, colonIndex).toInt();
    int angle = command.substring(colonIndex + 1).toInt();
    
    if (motorID >= 0 && motorID <= 15 && angle >= 0 && angle <= 120) {
        Serial.println("Moving Servo: " + String(motorID) + " to " + String(angle));
        int pwm_value = pulseWidth(angle);
        pwm.setPWM(motorID, 0, pwm_value);
    } else {
      Serial.println("Invalid command. Use format MOTOR_ID:ANGLE (e.g., 4:90)");
    }
  } else {
    Serial.println("Invalid command format. Use MOTOR_ID:ANGLE");
  }
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    processCommands(command);
  }
}
