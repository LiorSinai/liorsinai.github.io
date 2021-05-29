"use strict"

let doors_ul = document.getElementById("doors");

function randInt(upper){
    return Math.floor(Math.random() * upper);
}

let doors = {};
let mode = 0; //0: select door; 1: switch; 2: reset
let stats = {switched: {won: 0, lost: 0}, stayed: {won: 0, lost: 0}};

document.getElementById("num-doors").value = 3;
init();
reset();


function init(){
    let num_doors = +document.getElementById("num-doors").value;
    if (num_doors < 3){
        num_doors = 3;
        document.getElementById("num-doors").value = 3;
    }

    doors = {};
    mode = 0;
    stats = {switched: {won: 0, lost: 0}, stayed: {won: 0, lost: 0}};

    doors_ul.innerHTML = '';
    for (let i=1; i<=num_doors; i++){
        let entry = document.createElement('li');
        let div = document.createElement('div');
        div.className = "door";
        div.onclick = () => selectDoor(i);
        entry.appendChild(div);
        doors_ul.appendChild(entry);
    }

    reset();
    updateStats();
}

function reset(){
    mode = 0;
    let num_doors = +document.getElementById("num-doors").value;

    // reset doors object
    doors = {};
    for (let i=1; i<=num_doors; i++){
        doors[i] =  {closed: true, selected: false, goat: false}
    };
    doors.size = num_doors;

    // radomly select one to not be a goat
    let j = randInt(num_doors) + 1;
    for (let i=1; i <= doors.size; i++){
        if (i != j){
            doors[i].goat = true;
        }
    }

    // display
    for (let i=1; i <= doors.size; i++){
        let door = doors_ul.children[i - 1].children[0];
        door.style.borderColor = "black"
        door.className = "door";
        door.innerHTML = `<br> Door <br> ${i} <br>`    
    }
    let message = document.getElementById("message");
    message.innerHTML = "Pick a door.";
}

function selectDoor(id){
    if (mode == 2){
        reset();
        return;
    }
    else if (mode == 1){
        if (!doors[id].closed){
            return;
        }
        let action = doors[id].selected ? "stayed" : "switched";
        mode = 2;
        endGame(id, action);
    }
    else if (mode == 0){
        mode = 1;
        doors[id].selected = true;
        revealGoats(id);
    }

    // display
    doors_ul.children[id - 1].children[0].style.borderColor = "red";
    // reset other doors  
    for (let i=0; i < doors_ul.children.length; i++){
        if (i != id - 1) {doors_ul.children[i].children[0].style.borderColor  = "black"; }
    }
}

function revealGoats(selectedId){
    let carId = - 1;
    if (!doors[selectedId].goat){
        // first choice is correct
        // choose another random goat to show
        while (carId == -1){
            let j = randInt(doors.size) + 1;
            carId = (j == selectedId) ? -1 : j;  
        }
    }
    else{ // find non-goat. 
        // this branch is more likely and is at the heart of the Monty Hall controversy
        for (let i=1; i <= doors.size; i++){
            if (!doors[i].goat){
                carId = i;
                break;
            }
        }
    }
    // display goats
    for (let i=1; i <= doors.size; i++){
        if ((i != selectedId) && (i != carId)){
            let door = doors_ul.children[i-1].children[0];
            door.className += " goat";
            door.innerHTML = "<br>goat<br><br>";
            doors[i].closed = false;
        }
    }
    let message = document.getElementById("message");
    message.innerHTML = "Goat" + (doors.size > 3 ? "s" : "") + " revealed. Do you want to switch? Click on the same door to not switch.";
}

function endGame(selectedId, action){
    if (doors[selectedId].goat){
        stats[action].lost += 1;
    }
    else{
        stats[action].won += 1;
    }

    displayResults(selectedId);
}

function displayResults(selectedId){
    let selectedDoor = doors_ul.children[selectedId - 1].children[0];
    let message = document.getElementById("message");
    if (doors[selectedId].goat){
        selectedDoor.className += " goat";
        selectedDoor.innerHTML = "<br>goat<br><br>"
        message.innerHTML = "You got a goat.";
    }
    else {
        selectedDoor.className += " car";
        selectedDoor.innerHTML = "<br>car<br><br>";
        message.innerHTML = "You won a car!";
    }
    message.innerHTML += " Click on a door to try again."
    updateStats();
}

function updateStats(){
    let stayed_count = stats.stayed.won + stats.stayed.lost
    let switched_count = stats.switched.won + stats.switched.lost;

    document.getElementById("stayed_count").innerText = stayed_count;
    let x = stayed_count> 0 ? (stats.stayed.won/stayed_count): 0;
    document.getElementById("stayed_won").innerText = (x * 100).toFixed(1) + "%";
    
    document.getElementById("switched_count").innerText = switched_count;
    let y = switched_count > 0 ? (stats.switched.won/ switched_count): 0;
    document.getElementById("switched_won").innerText = (y * 100).toFixed(1) + "%";
}
 