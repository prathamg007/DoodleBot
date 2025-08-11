int pot1Pin=A1;
int pot2Pin=A2;

int buzzerPins[]={2,3,4};
int ledPins[]={5,6,7,8,9};

int ledIndex=0;
long lastUpdateTime=0;
int delayDuration=500;

void setup() {
  //buzzer pins
  for (int i=0; i<3; i++ ){
    pinMode(buzzerPins[i], OUTPUT);
    digitalWrite(buzzerPins[i], LOW);
  }

  //LEd pins
  for (int i=0; i<5; i++){
    pinMode(ledPins[i], OUTPUT);
    digitalWrite(ledPins[i], LOW);
  }

}

void loop() {
  //POT1
  int pot1Value= analogRead(pot1Pin);

  // turn off all buzzers
  for (int i=0; i<3; i++ ){
    digitalWrite(buzzerPins[i], LOW);
  }

  if (pot1Value <=341){
    digitalWrite(buzzerPins[0], HIGH);
  }else if (pot1Value>341 && pot1Value<=682){
    digitalWrite(buzzerPins[1], HIGH);
  } else{
    digitalWrite(buzzerPins[2], HIGH);
  }

  // POT2
  int pot2Value= analogRead(pot2Pin);
  delayDuration= map(pot2Value, 0, 1023, 100, 1000);

  long currentTime= millis();
  if (currentTime - lastUpdateTime >=delayDuration){
    lastUpdateTime= currentTime;
    
    //turn off the leds
    for (int i=0; i<5; i++){
      digitalWrite(ledPins[i], LOW);
    }

  //
    digitalWrite(ledPins[ledIndex], HIGH);

  //
    ledIndex= (ledIndex +1) % 5;
  }
}

//https://www.tinkercad.com/things/dPk2fZz3fP9/editel?lessonid=EFU6PEHIXGFUR1J&projectid=OGK4Q7VL20FZRV9&collectionid=undefined&title=Editing%20Components#/lesson-viewer:~:text=1%20result%20is%20available%2C%20use%20up%20and%20down%20arrow%20keys%20to%20navigate.
