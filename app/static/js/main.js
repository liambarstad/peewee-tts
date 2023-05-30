function submitQueryText(event) {
    event.preventDefault();  // prevent default form submission behavior
    const form = event.target;
    const url = form.action;
    const data = new FormData(form);

    const existingAudio = document.getElementById('audio-playback')
    if (existingAudio) { existingAudio.remove() }

    const existingError = document.getElementById('audio-error')
    if (existingError) { existingError.remove() }
  
    fetch(url, {
      method: 'POST',
      body: data
    })
    .then(response => response.blob())
    .then(blob => handleResponse(blob))
    .catch(error => handleResponse(null));
  }
  
  function handleResponse(blob) {
    const peeWeeForm = document.getElementById('peewee-form')
    const buttonEl = document.getElementById('speak-btn')
    if (blob) {
      const objectURL = URL.createObjectURL(blob);
      const audioElement = document.createElement('audio')
      audioElement.src = objectURL
      audioElement.controls = true
      audioElement.id = 'audio-playback'
      peeWeeForm.insertBefore(audioElement, buttonEl)
      URL.revokeObjectURL(blob)
    } else {
      const errorElement = document.createElement('p')
      errorElement.id = 'audio-error'
      errorElement.innerText = 'Sorry, something went wrong. Please try again later or email business@liambarstad.com'
      peeWeeForm.insertBefore(errorElement, buttonEl)
    }
  }

  function validateInput(event) {
    let btnEl = document.getElementById('speak-btn')
    if (event.target.value.length > 0 && event.target.value.length <= 100) {
      btnEl.disabled = false
    } else {
      btnEl.disabled = true
    }
  }