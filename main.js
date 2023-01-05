const choices = document.querySelectorAll('.choice');
const score = document.getElementById('score');
const result = document.getElementById('result');
const restart = document.getElementById('restart');
const modal = document.querySelector('.modal');
const past_score=document.querySelector('.past_score')
let control = 0;
let raunt=0;
const scoreboard = {
  player: 0,
  tie: 0,
  computer: 0
};

// Play game
function play(e) {
  control++;
  alert(control);
  restart.style.display = 'inline-block';
  const playerChoice = e.target.id;
  const computerChoice = getComputerChoice();
  const winner = getWinner(playerChoice, computerChoice);
  showWinner(winner, playerChoice, computerChoice);
  if (control === 10) {
    raunt++;
    console.log("raunt:"+raunt)
    console.log("player:" + scoreboard.player);
    console.log("computer:" + scoreboard.computer);
    control = 0;
    console.log("raunt bitti");
   const test=`
   Raunt: ${raunt}
   Player: ${scoreboard.player}
   Computer: ${scoreboard.computer}
 `
 const node = document.createElement("br");
 past_score.append(test)
 past_score.append(node)
 scoreboard.player = 0;
    scoreboard.tie = 0;
    scoreboard.computer = 0;
    score.innerHTML = `
    <p>Player: 0</p>
    <p>Tie: 0</p>
    <p>Computer: 0</p>
  `;
  }
}
function pastScore(){
  control++;
  alert(control)
  if (control === 10) {
    raunt++;
    console.log("raunt:"+raunt)
    console.log("player:" + scoreboard.player);
    console.log("computer:" + scoreboard.computer);
    control = 0;
    console.log("raunt bitti");
   const test=`
   Raunt: ${raunt}
   Player: ${scoreboard.player}
   Computer: ${scoreboard.computer}
 `
 const node = document.createElement("br");
 past_score.append(test)
 past_score.append(node)
 scoreboard.player = 0;
    scoreboard.tie = 0;
    scoreboard.computer = 0;
    score.innerHTML = `
    <p>Player: 0</p>
    <p>Tie: 0</p>
    <p>Computer: 0</p>
  `;
  }
}
// Get computers choice
function getComputerChoice() {
  const rand = Math.random();
  if (rand < 0.34) {
    return 'rock';
  } else if (rand <= 0.67) {
    return 'paper';
  } else {
    return 'scissors';
  }
}
// Get game winner
function getWinner(p, c) {
  if (p === c) {
    return 'draw';
  } else if (p === 'rock') {
    if (c === 'paper') {
      return 'computer';
    } else {
      return 'player';
    }
  } else if (p === 'paper') {
    if (c === 'scissors') {
      return 'computer';
    } else {
      return 'player';
    }
  } else if (p === 'scissors') {
    if (c === 'rock') {
      return 'computer';
    } else {
      return 'player';
    }
  }
}

function showWinner(winner, playerChoice, computerChoice) {
  if (winner === 'player') {
    // Inc player score
    scoreboard.player++;
    // Show modal result
    result.innerHTML = `
      <h1 class="text-win">Sen kazandın !</h1>
      <i class="fas fa-hand-${playerChoice} fa-10x"></i>
      <i class="fas fa-user fa-5x"></i>
      <p>Senin seçimin <strong>${playerChoice.charAt(0).toUpperCase() +
      playerChoice.slice(1)}</strong></p>
      <i class="fas fa-hand-${computerChoice} fa-10x"></i>
        <i class="fas fa-desktop fa-5x"></i>
      <p>Bilgisayarın seçimi <strong>${computerChoice.charAt(0).toUpperCase() +
      computerChoice.slice(1)}</strong></p>
    `;
  } else if (winner === 'computer') {
    // Inc computer score
    scoreboard.computer++;
    // Show modal result
    result.innerHTML = `
      <h1 class="text-lose">PC kazandı :(</h1>
     <i class="fas fa-hand-${playerChoice} fa-10x"></i>
      <i class="fas fa-user fa-5x"></i>
      <p>Senin seçimin <strong>${playerChoice.charAt(0).toUpperCase() +
      playerChoice.slice(1)}</strong></p>
      <i class="fas fa-hand-${computerChoice} fa-10x"></i>
        <i class="fas fa-desktop fa-5x"></i>
      <p>Bilgisayarın seçimi <strong>${computerChoice.charAt(0).toUpperCase() +
      computerChoice.slice(1)}</strong></p>
    `;
  } else {
    scoreboard.tie++;
    result.innerHTML = `
      <h1>Berabere</h1>
      <i class="fas fa-hand-${playerChoice} fa-10x"></i>
      <i class="fas fa-user fa-5x"></i>
      <p>Senin seçimin <strong>${playerChoice.charAt(0).toUpperCase() +
      playerChoice.slice(1)}</strong></p>
      <i class="fas fa-hand-${computerChoice} fa-10x"></i>
        <i class="fas fa-desktop fa-5x"></i>
      <p>Bilgisayarın seçimi <strong>${computerChoice.charAt(0).toUpperCase() +
      computerChoice.slice(1)}</strong></p>
    `;
  }
  // Show score
  score.innerHTML = `
    <p>Player: ${scoreboard.player}</p>
    <p>Tie: ${scoreboard.tie}</p>
    <p>Computer: ${scoreboard.computer}</p>
    `;

  modal.style.display = 'block';
}

// Restart game
function restartGame() {
  scoreboard.player = 0;
  scoreboard.tie = 0;
  scoreboard.computer = 0;
  score.innerHTML = `
    <p>Player: 0</p>
    <p>Tie: 0</p>
    <p>Computer: 0</p>
  `;
}

// Clear modal
function clearModal(e) {
  if (e.target == modal) {
    modal.style.display = 'none';
  }
}

// Event listeners
choices.forEach(choice => choice.addEventListener('click', play));
document.addEventListener("keyup", function (event) {
  restart.style.display = 'inline-block';

  let playerKey = "";

  if (event.key === 'a') {
    playerKey = 'rock';
  }
  else if (event.key === 'b') {
    playerKey = 'paper';
  }
  else if (event.key === 'c') {
    playerKey = 'scissors'
  }
  if (['rock', 'paper', 'scissors'].includes(playerKey)) {
    const computerChoice = getComputerChoice();
    const winner = getWinner(playerKey, computerChoice);
    showWinner(winner, playerKey, computerChoice);
    pastScore()
    //falösalda
  }
});
window.addEventListener('click', clearModal);
restart.addEventListener('click', restartGame);
