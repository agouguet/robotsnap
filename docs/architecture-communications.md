# RobotSNAP · architecture des communications

Schéma : [`architecture-communications.svg`](architecture-communications.svg) (source vectorielle, éditable, autonome) et [`architecture-communications.png`](architecture-communications.png) (aperçu rendu).

## Périmètre et méthode

Note établie par **lecture statique** du code actif le **2 octobre 2026** : côté Python `src/robotsnap/`, `README.md` et `docs/` de ce dépôt ; côté Unity les scripts `Assets/Scripts/RobotSNAP/` et le paquet `com.unity.robotics.ros-tcp-connector` du projet Unity (`/home/adam/robotsnap-unity`). Aucune session Unity ni aucun test réseau ou mesure de performance n'ont été exécutés lors de cette analyse : le document décrit ce que le code **déclare**, pas un comportement observé en session.

## Intention

**Python décide, Unity simule, les transports acheminent.** Un seul endpoint Unity, deux façons de l'atteindre : A (Python pur, sans ROS) ou B (ROS 2 natif). Les chemins A et B sont **alternatifs** et ne coexistent pas sur le port 10000.

## Responsabilités

- **Python — décisions et orchestration.** Algorithmes de navigation, entraînement / inférence RL (Gymnasium, PyTorch), observations et récompenses, viewer et analyse. La façade est `RobotSNAPClient`, ouverte par `open_client(transport="tcp")` ou `open_client(transport="ros2")`.
- **Unity — simulateur.** Monde et scénarios, robots et piétons, physique et temps de simulation, capteurs (lidar, odométrie), enregistrement des métriques d'épisode. Unity **exécute** les commandes reçues et **publie** observations et résultats.

## Un seul endpoint, un seul protocole filaire

- Unity est **client TCP** : il ouvre la connexion vers un **serveur** Python, par défaut `127.0.0.1:10000`. Le sens d'ouverture de la connexion (Unity → serveur) est **distinct** du sens des messages : les commandes vont vers Unity, les observations et réponses reviennent vers Python.
- Un **seul** serveur écoute sur ce port : soit le pont Python (chemin A), soit `ros_tcp_endpoint` (chemin B), jamais les deux.
- Le format est le protocole **ROS-TCP-Connector** avec messages ROS 2 sérialisés en **CDR**. Le TCP sans ROS utilise **exactement** ce même format.
- **Mise au point CDR :** les charges JSON voyagent dans des `std_msgs/String`, elles-mêmes sérialisées en CDR. Le pont sans ROS utilise `rosbags` ; `rclpy` intervient dans les accès ROS 2.

## Chemin A — sans ROS (défaut)

`RobotSNAPClient (process Python)` ↔ `RobotSNAPBridge (serveur TCP)` ↔ `TCP :10000` ↔ `Unity · ROS-TCP-Connector`.

- Client et pont communiquent **en mémoire, dans le même processus** : pas de second réseau ni de second serveur.
- Le pont gère la connexion, les types, l'encodage / décodage CDR via `rosbags`, le dernier état par topic et l'acheminement des commandes.
- Aucune installation ROS n'est requise.

### Extension optionnelle du chemin A

`RobotSNAPBridge` ↔ `Ros2Gateway` ↔ `graphe ROS 2 / DDS` ↔ `nœuds ROS 2` (navigation, outils, `rclpy`, éventuellement Docker).

- Lancée par `python -m robotsnap.bridge --ros2` ; Unity se connecte **toujours** au pont Python.
- C'est un **miroir bidirectionnel** des publications et abonnements déclarés par Unity : `Ros2Gateway` republie le CDR désérialisé sans copier les champs, saute les types inconnus et n'a pas de file d'attente.
- **`ros_tcp_endpoint` n'est pas utilisé** dans ce chemin.

## Chemin B — ROS 2 natif (alternatif)

`Python (Ros2Client / rclpy) + autres nœuds ROS 2` ↔ `graphe ROS 2 (DDS)` ↔ `ros_tcp_endpoint (serveur TCP alternatif)` ↔ `TCP :10000` ↔ `Unity`.

- Le client Python rejoint le graphe ROS 2 par DDS : il **ne prend pas** le port 10000.
- `ros_tcp_endpoint` tient le port 10000 et relaie TCP ↔ ROS 2 ; ROS 2 distribue les topics aux nœuds.
- Unity reste la simulation ; la connexion :10000 est ouverte par Unity vers le serveur.

## Messages communs à tous les chemins

| Topic | Type | Sens | Rôle |
|---|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | Python → Unity | vitesse linéaire / angulaire |
| `/simulation/control` | `std_msgs/String` (JSON) | Python → Unity | play, pause, reset, scénario, objectifs, piétons, temps / pas |
| `/odom` | `nav_msgs/Odometry` | Unity → Python | pose et twist du robot |
| `/scan` | `sensor_msgs/LaserScan` | Unity → Python | lidar |
| `/map` | `nav_msgs/OccupancyGrid` | Unity → Python | carte d'occupation |
| `/simulation/state` | `std_msgs/String` (JSON) | Unity → Python | état de session |
| `/simulation/agents` | `std_msgs/String` (JSON) | Unity → Python | robots / piétons, dans le repère robot |
| `/simulation/control_result` | `std_msgs/String` (JSON) | Unity → Python | acquittement d'une commande |
| `/reset_done` | `std_msgs/Bool` | Unity → Python | monde prêt |
| `/simulation/metrics` | `std_msgs/String` (JSON) | Unity → Python | bilan d'épisode (additif) |

`/clock` (`rosgraph_msgs/Clock`) peut publier l'horloge s'il est activé ; il est omis du schéma pour la lisibilité.

## Notes discrètes

- **Multi-robot.** Le robot principal `robot_1` conserve les topics nus `/cmd_vel`, `/odom`, `/scan` ; les autres robots publient et écoutent sous `/robot_<id>/...`. Les topics globaux, comme `/simulation/state` et `/simulation/control`, restent communs à la session ; `/simulation/agents` suit chaque robot (par exemple `/robot_2/simulation/agents`).
- **Scénarios.** Les scénarios YAML sont écrits **sur disque**, puis Unity reçoit leur **nom** via la commande `load_scenario` : ce n'est pas un transfert YAML sur TCP.
- **Avant abonnement.** Une commande envoyée avant que le pair ne se soit abonné est perdue ; le serveur envoie la trame `__handshake` (protocole `ROS2`) avant les messages applicatifs.

## Références (code actif)

**Python — dépôt `/home/adam/robotSNAP_ws` :**

- `README.md:7-10` — Unity client TCP, serveur Python côté Python ; protocole identique.
- `src/robotsnap/bridge/server.py:160-179` — `RobotSNAPBridge`, hôte `0.0.0.0`, port `10000` par défaut.
- `src/robotsnap/bridge/server.py:457-470` — trame `__handshake` envoyée en premier.
- `src/robotsnap/bridge/protocol.py:1-11` — format de trame `[len][destination][size][payload]`.
- `src/robotsnap/bridge/codec.py:1-3` et `:36` — encodage / décodage CDR via `rosbags` (typestore ROS 2 Humble).
- `robotsnap-ros2/src/robotsnap_ros2/gateway.py:1-22` et `:45` — miroir bidirectionnel `Ros2Gateway`.
- `src/robotsnap/bridge/__main__.py:38` et `:48` — port par défaut et option `--ros2`.
- `src/robotsnap/client.py:1152-1182` — façade `open_client(transport="tcp")` ou `open_client(transport="ros2")`.
- `robotsnap-ros2/src/robotsnap_ros2/support.py:1-10` — `Ros2Client` sur graphe ROS 2 via `rclpy`, sans socket ni `ros_tcp_endpoint` (le cœur ne le connaît que par le point d'entrée `robotsnap.transports`).
- `src/robotsnap/topics.py:90-101` — table `TOPIC_TYPES` ; `:121-135` — `robot_topic` (`/robot_<id>/...`).
- `src/robotsnap/analysis/metrics.py:74` — constante du topic `/simulation/metrics`.
- `docs/robotsnap-unity-contract.md:9-21`, `:35-46`, `:474-498` — contrat filaire, table des topics, multi-robot.
- `robotsnap-ros2/docs/ros2-with-a-docker-method.md:7-24`, `:107-131` — usage de `ros_tcp_endpoint` côté ROS 2.

**Unity — projet `/home/adam/robotsnap-unity` :**

- `Library/PackageCache/com.unity.robotics.ros-tcp-connector@bdad6c87bd9e/Runtime/TcpConnector/ROSConnection.cs:25-32` — défaut `127.0.0.1:10000`.
- `.../ROSConnection.cs:789` — `TcpClient` : Unity se connecte comme client ; `:653-660` — vérification du protocole `ROS2`.
- `Assets/Scripts/RobotSNAP/ROS/EnvROS.cs:169-171` — pair interchangeable `ros_tcp_endpoint` ou serveur Python ; `:238-246` — publication ignorée si aucun publisher n'est enregistré.
- `Assets/Scripts/RobotSNAP/ROS/RobotSNAPTopics.cs:38` et `:127-142` — contrat des topics, dont `/cmd_vel` et `/reset_done`.
- `Assets/Scripts/RobotSNAP/Agent/Robot/RobotIdentity.cs:13-20` et `:93-97` — `robot_1` garde les topics nus, les autres en `/robot_<id>/...`.
- `Assets/Scripts/RobotSNAP/ROS/SimulationControlBridge.cs:210` et `:323` — abonnement à `/simulation/control`, publication de `/simulation/control_result`.
- `Assets/Scripts/RobotSNAP/Agent/Robot/RobotInputController.cs:284` et `:288` — abonnement et traitement de `/cmd_vel`.
- `ProjectSettings/ProjectSettings.asset:852` — le projet compile en mode `ROS2`.

## Limites

- Analyse statique : aucun test live, aucune mesure de débit ni de latence.
- `ros_tcp_endpoint` est **hors de ce dépôt** ; son comportement à l'exécution n'a pas été testé lors de cette analyse.
- L'extension `--ros2` et le chemin B ne sont pas validés en session réelle dans cette note.
