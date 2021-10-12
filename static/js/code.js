pred_btn = document.querySelector('#pred_btn')

dataset = document.getElementById('datasets') // current selected dataset
dataset_options = document.querySelectorAll('#datasets option') // datasets list

model = document.getElementById('models') // current selected model

pred_txt = document.getElementById('pred_txt') // label that show the predicted count
pred_img = document.getElementById('pred_img') // predicted density map

slider = document.getElementById('slider') // slider component
slider_imgs = document.querySelectorAll('#slider img') // current selected dataset images

cur_img = document.getElementById('curr_image') // current selected image
gt_count = document.getElementById("curr_txt") // label that show the ground truth


// ip address (define without last slash)
url_address = ''

// dropdown initialization
dataset.selelectedIndex = 0
model.selelectedIndex = 0

// slider initialization
default_img = url_address + '/static/default.png'
default_img_name = 'none'

cur_img.src = default_img
cur_img.setAttribute('img_name', default_img_name)

last_path = cur_img.src
last_img_name = cur_img.getAttribute('img_name')


const updateCurrImage = image => {

    if (last_path === image.src) {
        cur_img.src = default_img
        cur_img.setAttribute('img_name', default_img_name)
    }
    else {
        cur_img.src = image.src
        cur_img.setAttribute('img_name', image.getAttribute('img_name'))
    }

    pred_img.src = default_img;
    pred_txt.innerHTML = 0;

    last_path = cur_img.src;
    last_img_name = cur_img.getAttribute('img_name')
}

const getGroundtruth = () => {
    if (cur_img.getAttribute('img_name') !== 'none') {
        fetch(`${url_address}/groundtruth/${cur_img.getAttribute('img_name')}/?dataset=${dataset.value}`)
        .then(res => res.json())
        .then(res => {
            gt_count.innerHTML =  res.human_num
        })
        .catch(err => console.log(err))
    } else {
        gt_count.innerHTML = 0
    }
}

const generateImages = (images) => {
    // remove all child inside slider
    while (slider.firstChild) {
        slider.removeChild(slider.lastChild)
    }
    images.forEach((img, index) => {
        let element = document.createElement('img')
        element.setAttribute('img_name', images[index].name)
        element.src = images[index].path
        element.addEventListener('click', event => {
            const image = event.target
            updateCurrImage(image)
            getGroundtruth()
        })
        slider.appendChild(element)// append the element inside the component
    })
}

const updateSlider = () => {
    fetch(`${url_address}/images/?dataset=${dataset.value}`)
    .then(res => res.json())
    .then(images => {
        generateImages(images)
        //getGroundtruth() // why???
    })
    .catch(err => console.log(err))
}

const predict = () => {
    if (
        !dataset.value || cur_img.src === default_img || !model.value
    ) {
        console.log('Errori nella compilazione!')
        return
    }

    _image = /\/static.*/.exec(cur_img.src)[0];

    data = {
        dataset: dataset.value,
        image: _image,
        model: model.value,
    }

    options = {
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        method: 'post',
        body: JSON.stringify(data)
    }

    fetch(`${url_address}/predict`, options)
        .then(res => res.json())
        .then(res => {
            pred_txt.innerHTML = res
            pred_img.src = url_address + '/static/map.jpg'
            /*window.location.href = "http://localhost:5000/predict/" + res;*/
        })
        .catch(err => console.log(err))
}


// update images when the page is been loaded
window.addEventListener("load",  () => {
    console.log('loading...')
    updateSlider()
})

datasets.addEventListener('change', e => {
    updateSlider()
})

model.addEventListener('change', e => {
    pred_img.src = default_img;
    pred_txt.innerHTML = 0;
})

pred_btn.onclick = e => {
    predict()
}
