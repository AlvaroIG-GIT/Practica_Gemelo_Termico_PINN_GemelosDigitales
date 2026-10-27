# main.py — ESP32 (MicroPython) · barra térmica: 5 x DS18B20 + resistencia por MOSFET, publicación MQTT
# Cableado: DS18B20 en bus 1-Wire (GPIO4, pull-up 4k7); MOSFET del calefactor en GPIO16; LED en GPIO2.
# Topics: lab/barra/<PAREJA>/temp (JSON cada 2 s) · lab/barra/<PAREJA>/cmd ("on"/"off"/"ensayo")
import time, json, network, machine, onewire, ds18x20
from umqtt.simple import MQTTClient

WIFI_SSID, WIFI_PASS = "LAB_UIE", "********"
BROKER, PAREJA = "192.168.1.10", "p01"
PERIODO_S, T_ON_S, T_END_S = 2.0, 300, 900
ORDEN_SENSORES = []          # ROMs de S1..S5 en orden de x (3, 9, 15, 21, 27 cm); vacío = orden de descubrimiento

ow = ds18x20.DS18X20(onewire.OneWire(machine.Pin(4)))
roms = ORDEN_SENSORES or ow.scan()
calef = machine.Pin(16, machine.Pin.OUT, value=0); led = machine.Pin(2, machine.Pin.OUT)

wlan = network.WLAN(network.STA_IF); wlan.active(True); wlan.connect(WIFI_SSID, WIFI_PASS)
while not wlan.isconnected(): time.sleep(0.5)
t_ensayo = None
def on_cmd(topic, msg):
    global t_ensayo
    m = msg.decode()
    if m == "on": calef.value(1)
    elif m == "off": calef.value(0); t_ensayo = None
    elif m == "ensayo": t_ensayo = time.ticks_ms(); calef.value(1)      # pulso automático de T_ON_S
cli = MQTTClient(f"barra-{PAREJA}", BROKER); cli.set_callback(on_cmd); cli.connect()
cli.subscribe(f"lab/barra/{PAREJA}/cmd".encode())
cli.publish(f"lab/barra/{PAREJA}/info", json.dumps({"sensores": len(roms), "x_cm": [3, 9, 15, 21, 27], "P_W": 2.5, "L_cm": 30}), retain=True)

while True:
    cli.check_msg()
    ow.convert_temp(); time.sleep_ms(750)
    temps = [round(ow.read_temp(r), 2) for r in roms]
    t_rel = None
    if t_ensayo is not None:
        t_rel = time.ticks_diff(time.ticks_ms(), t_ensayo) / 1000
        if t_rel >= T_ON_S: calef.value(0)
        if t_rel >= T_END_S: t_ensayo = None
    msg = {"t_s": t_rel, "T": temps, "calef": calef.value(), "ts": time.time()}
    cli.publish(f"lab/barra/{PAREJA}/temp", json.dumps(msg)); led.value(not led.value())
    time.sleep(PERIODO_S - 0.75)
