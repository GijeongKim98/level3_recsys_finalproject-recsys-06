let BIG_TAGS = ["basics", "hash", "recursion", "search", "math",
                "data structure", "tree", "graph", "dp", "greedy", "bruteforce"];

function set_title(user_name) {
    const heading = document.getElementById("home_title");
    heading.textContent =  `${user_name}님, 반갑습니다!`;
    document.title = `${user_name}님의 Home`;
}

function draw_stat_graph(canvas_id, bar_data, line_data) {
    const ctx = document.getElementById(canvas_id);
    new Chart(ctx, {
        data: {
        labels: BIG_TAGS,
        datasets: [{
            type: 'bar',
            data: bar_data,
            borderWidth: 1,
            backgroundColor: "#32c79b88", 
            borderColor: "#32c79bdd", 
        }, {
            type: 'line',
            data: line_data,
            backgroundColor: "#35CB85C88", 
            borderColor: "#5CB85Cdd", 
        }]
        },
        options: {
        scales: {
            y: {
            beginAtZero: true
            },
        },
        plugins: {
            legend: {
               display: false,
            }
        },
        responsive: true,
        maintainAspectRatio: false,
        }
    });
}

function clear_problem_list() {
    const item_problems = document.getElementsByClassName("item_problem");
    while (item_problems.length != 0) {
        item_problems[0].remove();
    }
}

function add_problem_list(id, title, tags) {
    const __problem_card = document.getElementById("test_card");
    const _problem_card = __problem_card.getElementsByClassName("app_card")[0];
    const content = _problem_card.getElementsByClassName("app_card_table")[0];
    const str = `
    <div class="app_card_table_item item_problem">
        <div class="item_text">${id}</div>
        <div class="item_text">${title}</div>
        <div class="item_tags"></div>
    </div>
    `
    content.insertAdjacentHTML('beforeend', str);

    const tag_divs = content.getElementsByClassName("item_problem");
    const last_tag_divs = tag_divs[tag_divs.length -1];
    const tag_div = last_tag_divs.getElementsByClassName("item_tags")[0];
    for (i in tags) {
        const str = `<div class="item_tag">${tags[i]}</div>`;
        tag_div.insertAdjacentHTML('beforeend', str);
    }
}

function clear_repo_list() {
    const item_repos = document.getElementsByClassName("item_repo");
    while (item_repos.length != 0) {
        item_repos[0].remove();
    }
}

function add_repo_list(title, description, domain) {
    const __repo_card = document.getElementById("repo_card");
    console.log(__repo_card)
    const content = __repo_card.getElementsByClassName("app_card")[0];
    console.log(content)
    const str = `
    <div class="app_card_table_item item_repo">
        <div class="item_text">${title}</div>
        <div class="item_text">${description}</div>
        <div class="item_text">${domain}</div>
    </div>
    `
    content.insertAdjacentHTML('beforeend', str);
}

window.onload = function() {
    const user = localStorage.getItem("user");
    set_title(user);
    const url = `https://8290-49-50-165-98.ngrok-free.app/user/${user}/btag`

    // const btag = fetch(url, {
    //     method: "get",
    //     headers: new Headers({
    //     "ngrok-skip-browser-warning": "69420",
    //     }),
    // }).then((response) => response.json())
    // .then((data) => console.log(data));

    const btag = {
        "user_id":"kkj9818","tree":10.729559938252086,"data structure":880.0459839506456,"graph":944.770312728214,
        "hash":127.34921184162762,"search":165.79476614862824,"dp":435.3592561963338,"greedy":198.4281883655579,
        "math":340.02889057170825,"bruteforce":212.4184219183956,"recursion":346.85322252739985,"basics":706.3678163790012};
    const user_data = []
    for (var i = 0; i < BIG_TAGS.length; i++) {
        user_data.push(parseInt(btag[BIG_TAGS[i]]));
    };
    const naver_btag = {"corporation_name": '네이버', "tree": 68.91456993216241, 
    "data structure": 722.4625808393837, "graph": 959.5836232973572, "hash": 94.30426232610468, 
    "search": 122.77646562648036, "dp": 446.5599599053222, "greedy": 158.3769285220893, 
    "math": 336.9676416169149, "bruteforce": 358.5021229819837, "recursion": 149.80522412747382, "basics": 969.2359432237768}
    const naver_data = [];
    for (var i = 0; i < BIG_TAGS.length; i++) {
        naver_data.push(parseInt(naver_btag[BIG_TAGS[i]]));
    };
    draw_stat_graph("big_stat", user_data, naver_data);

    clear_problem_list();
    problem_list = [
        {"problem_id":"4779","level":8,"title":"칸토어 집합","bigtag":["recursion"],"midtag":["divide_and_conquer","recursion"]},
        {"problem_id":"15565","level":10,"title":"귀여운 라이언","bigtag":["data structure"],"midtag":["segtree","two_pointer"]},
        {"problem_id":"13422","level":12,"title":"도둑","bigtag":["data structure"],"midtag":["segtree","two_pointer"]},
        {"problem_id":"14438","level":15,"title":"수열과 쿼리 17","bigtag":["data structure"],"midtag":["segtree"]},
        {"problem_id":"14428","level":15,"title":"수열과 쿼리 16","bigtag":["data structure"],"midtag":["segtree"]},
        {"problem_id":"14235","level":8,"title":"크리스마스 선물","bigtag":["data structure"],"midtag":["priority_queue","segtree"]},
        {"problem_id":"14245","level":17,"title":"XOR","bigtag":["data structure"],"midtag":["segtree"]},
        {"problem_id":"14246","level":9,"title":"K보다 큰 구간","bigtag":["data structure"],"midtag":["segtree","two_pointer"]},
        {"problem_id":"12895","level":18,"title":"화려한 마을","bigtag":["data structure"],"midtag":["segtree"]},
        {"problem_id":"12899","level":17,"title":"데이터 구조","bigtag":["data structure"],"midtag":["segtree"]}
    ];

    for (i in problem_list) {
        const problem = problem_list[i];
        add_problem_list(problem["problem_id"], problem["title"], problem["bigtag"]);
    };

    clear_repo_list();
    const repo_list = [{"title":"1adrianb/face-alignment","tags":"['python',,'deep-learning',,'pytorch',,'face-detector',,'face-detection',,'face-alignment'","about":"🔥 2D and 3D Face alignment library build using pytorch","readme":"Detect facial landmarks from Python using the world's most accurate face alignment network, capable of detecting points in both 2D and 3D coordinates.","lang":"Python","tsne_text_x":0.040468633,"tsne_text_y":0.52881765,"label_28":"ML"},{"title":"aamini/introtodeeplearning","tags":"['mit',,'computer-vision',,'deep-learning',,'tensorflow',,'deep-reinforcement-learning',,'neural-networks',,'tensorflow-tutorials',,'deeplearning',,'jupyter-notebooks',,'music-generation',,'algorithmic-bias'","about":"Lab Materials for MIT 6.S191: Introduction to Deep Learning","readme":"This repository contains all of the code and software labs for MIT Introduction to Deep Learning! All lecture slides and videos are available on the program website.","lang":"Jupyter Notebook","tsne_text_x":0.1574567,"tsne_text_y":0.47435185,"label_28":"ML"},{"title":"adamian98/pulse","tags":"[","about":"PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models","readme":"Code accompanying CVPR'20 paper of the same title. Paper link: https://arxiv.org/pdf/2003.03808.pdf","lang":"Python","tsne_text_x":0.040878713,"tsne_text_y":0.4636291,"label_28":"ML"},{"title":"ageitgey/face_recognition","tags":"['python',,'machine-learning',,'face-recognition',,'face-detection'","about":"The world's simplest facial recognition api for Python and the command line","readme":"You can also read a translated version of this file in Chinese 简体中文版 or in Korean 한국어 or in Japanese 日本語.","lang":"Python","tsne_text_x":0.04291451,"tsne_text_y":0.52990687,"label_28":"ML"},{"title":"ageron/handson-ml","tags":"['python',,'machine-learning',,'deep-learning',,'neural-network',,'tensorflow',,'scikit-learn',,'jupyter-notebook',,'ml',,'deprecated',,'distributed'","about":"⛔️ DEPRECATED – See https://github.com/ageron/handson-ml3 instead.","readme":"This project is for the first edition, which is now outdated.","lang":"Jupyter Notebook","tsne_text_x":0.17594403,"tsne_text_y":0.4962717,"label_28":"ML"},{"title":"ageron/handson-ml2","tags":"[","about":"A series of Jupyter notebooks that walk you through the fundamentals of Machine Learning and Deep Learning in Python using Scikit-Learn, Keras and TensorFlow 2.","readme":"This project aims at teaching you the fundamentals of Machine Learning in\npython. It contains the example code and solutions to the exercises in the second edition of my O'Reilly book Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow:","lang":"Jupyter Notebook","tsne_text_x":0.17835063,"tsne_text_y":0.49167278,"label_28":"ML"},{"title":"AI4Finance-Foundation/FinRL","tags":"['finance',,'deep-reinforcement-learning',,'openai-gym',,'fintech',,'algorithmic-trading',,'stock-trading',,'multi-agent-learning',,'stock-markets',,'pythorch',,'tensorflow2',,'drl-trading-agents',,'drl-algorithms',,'drl-framework',,'trading-tasks'","about":"FinRL: Financial Reinforcement Learning. 🔥","readme":"FinGPT: Open-source for open-finance! Revolutionize FinTech.","lang":"Jupyter Notebook","tsne_text_x":0.17380342,"tsne_text_y":0.3947404,"label_28":"ML"},{"title":"AlexeyAB/darknet","tags":"['deep-neural-networks',,'computer-vision',,'deep-learning',,'neural-network',,'dnn',,'yolo',,'object-detection',,'deep-learning-tutorial',,'yolov3',,'yolov4',,'scaledyolov4',,'scaled-yolov4'","about":"YOLOv4 / Scaled-YOLOv4 / YOLO - Neural Networks for Object Detection (Windows and Linux version of Darknet )","readme":"Paper YOLOv7: https://arxiv.org/abs/2207.02696","lang":"C","tsne_text_x":0.08852151,"tsne_text_y":0.5009909,"label_28":"ML"},{"title":"alexeygrigorev/mlbookcamp-code","tags":"[","about":"The code from the Machine Learning Bookcamp book and a free course based on the book","readme":"The code from the Machine Learning Bookcamp book","lang":"Jupyter Notebook","tsne_text_x":0.1902343,"tsne_text_y":0.4805768,"label_28":"ML"},{"title":"alexjc/neural-doodle","tags":"['deep-neural-networks',,'deep-learning',,'image-processing',,'image-manipulation',,'image-generation'","about":"Turn your two-bit doodles into fine artworks with deep neural networks, generate seamless textures from photos, transfer style from one image to another, perform example-based upscaling, but wait... there's more! (An implementation of Semantic Style Transfer.)","readme":"Use a deep neural network to borrow the skills of real artists and turn your two-bit doodles into masterpieces! This project is an implementation of Semantic Style Transfer (Champandard, 2016), based on the Neural Patches algorithm (Li, 2016). Read more about the motivation in this in-depth article and watch this workflow video for inspiration.","lang":"Python","tsne_text_x":0.044411123,"tsne_text_y":0.45309222,"label_28":"ML"},{"title":"alexjc/neural-enhance","tags":"[","about":"Super Resolution for images using deep learning.","readme":"Example #1 — Old Station: view comparison in 24-bit HD, original photo CC-BY-SA @siv-athens.","lang":"Python","tsne_text_x":0.03660947,"tsne_text_y":0.48116753,"label_28":"ML"},{"title":"alibaba/MNN","tags":"['machine-learning',,'arm',,'deep-neural-networks',,'deep-learning',,'vulkan',,'ml',,'convolution',,'embedded-devices',,'mnn',,'winograd-algorithm'","about":"MNN is a blazing fast, lightweight deep learning framework, battle-tested by business-critical use cases in Alibaba","readme":"中文版本","lang":"C++","tsne_text_x":0.11342022,"tsne_text_y":0.4555594,"label_28":"ML"},{"title":"amueller/introduction_to_ml_with_python","tags":"[","about":"Notebooks and code for the book \"Introduction to Machine Learning with Python\"","readme":"This repository holds the code for the forthcoming book \"Introduction to Machine\nLearning with Python\" by Andreas Mueller and Sarah Guido.\nYou can find details about the book on the O'Reilly website.","lang":"Jupyter Notebook","tsne_text_x":0.18475118,"tsne_text_y":0.4922514,"label_28":"ML"},{"title":"apache/mxnet","tags":"['mxnet'","about":"Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with Dynamic, Mutation-aware Dataflow Dep Scheduler; for Python, R, Julia, Scala, Go, Javascript and more","readme":"Apache MXNet is a deep learning framework designed for both efficiency and flexibility.\nIt allows you to mix symbolic and imperative programming\nto maximize efficiency and productivity.\nAt its core, MXNet contains a dynamic dependency scheduler that automatically parallelizes both symbolic and imperative operations on the fly.\nA graph optimization layer on top of that makes symbolic execution fast and memory efficient.\nMXNet is portable and lightweight, scalable to many GPUs and machines.","lang":"C++","tsne_text_x":0.119697034,"tsne_text_y":0.46436164,"label_28":"ML"},{"title":"apache/tvm","tags":"['javascript',,'machine-learning',,'performance',,'deep-learning',,'metal',,'compiler',,'gpu',,'vulkan',,'opencl',,'tensor',,'spirv',,'rocm',,'tvm'","about":"Open deep learning compiler stack for cpu, gpu and specialized accelerators","readme":"Documentation |\nContributors |\nCommunity |\nRelease Notes","lang":"Python","tsne_text_x":0.13437602,"tsne_text_y":0.45726958,"label_28":"ML"},{"title":"apple/turicreate","tags":"['python',,'machine-learning',,'deep-learning'","about":"Turi Create simplifies the development of custom machine learning models.","readme":"Quick Links: Installation | Documentation","lang":"C++","tsne_text_x":0.14936921,"tsne_text_y":0.44784796,"label_28":"ML"},{"title":"arogozhnikov/einops","tags":"['deep-learning',,'chainer',,'tensorflow',,'numpy',,'keras',,'pytorch',,'tensor',,'cupy',,'jax',,'einops'","about":"Deep learning operations reinvented (for pytorch, tensorflow, jax and others)","readme":"Flexible and powerful tensor operations for readable and reliable code. \nSupports numpy, pytorch, tensorflow, jax, and others.","lang":"Python","tsne_text_x":0.12598419,"tsne_text_y":0.480655,"label_28":"ML"},{"title":"Atcold/pytorch-Deep-Learning","tags":"['deep-learning',,'jupyter-notebook',,'pytorch',,'neural-nets'","about":"Deep Learning (with PyTorch)","readme":"This notebook repository now has a companion website, where all the course material can be found in video and textual format.","lang":"Jupyter Notebook","tsne_text_x":0.16293734,"tsne_text_y":0.4887207,"label_28":"ML"},{"title":"automl/auto-sklearn","tags":"['scikit-learn',,'hyperparameter-optimization',,'bayesian-optimization',,'hyperparameter-tuning',,'automl',,'automated-machine-learning',,'smac',,'meta-learning',,'hyperparameter-search',,'metalearning'","about":"Automated Machine Learning with scikit-learn","readme":"auto-sklearn is an automated machine learning toolkit and a drop-in replacement for a scikit-learn estimator.","lang":"Python","tsne_text_x":0.1621356,"tsne_text_y":0.45066732,"label_28":"ML"},{"title":"aws/amazon-sagemaker-examples","tags":"['training',,'aws',,'data-science',,'machine-learning',,'reinforcement-learning',,'deep-learning',,'examples',,'jupyter-notebook',,'inference',,'sagemaker',,'mlops'","about":"Example 📓 Jupyter notebooks that demonstrate how to build, train, and deploy machine learning models using 🧠 Amazon SageMaker.","readme":"Example Jupyter notebooks that demonstrate how to build, train, and deploy machine learning models using Amazon SageMaker.","lang":"Jupyter Notebook","tsne_text_x":0.17914686,"tsne_text_y":0.4795124,"label_28":"ML"}];
    for (var i = 6; i < 11; i++) {
        console.log(i);
        const repo = repo_list[i]
        add_repo_list(repo["title"], repo["about"], "ml");
    };
};
