function submitQueryText(event) {
    event.preventDefault();  // prevent default form submission behavior
    const form = event.target;
    const url = form.action;
    const data = new FormData(form);
  
    fetch(url, {
      method: 'POST',
      body: data
    })
    .then(response => response.json())
    .then(data => handleResponse(data))
    .catch(error => console.error(error));
  }
  
  function handleResponse(data) {
    const message = data.message;
    const messageDiv = document.getElementById('message');
    messageDiv.innerText = message;
  }