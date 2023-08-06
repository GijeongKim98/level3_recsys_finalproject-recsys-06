let BIG_TAGS = [
    "basics", 
    "hash", 
    "recursion", 
    "search", 
    "math",
    "data structure", 
    "tree", 
    "graph", 
    "dp", 
    "greedy", 
    "bruteforce"
];


let MIDDLE_TAGS = [
    "implementation", "string", "sorting",
    "tree set", "hashing",
    "recursion", "divide and conquer",
    "binary search", "ternary search", "parametric search",
    "math",
    "stack", "queue", "priority queue", "deque", "disjoint set", "two pointer", "segtree",
    "trees",
    "graphs", "bfs", "dfs",
    "dp",
    "greedy",
    "bruteforcing", "mitm"
];

let TAG_LEVEL_MASK = [
    3,
    2,
    2,
    3,
    1,
    7,
    1,
    3,
    1,
    1,
    2
];

let TAG_LEVEL_BIG = 0;
let TAG_LEVEL_MIDDLE = 1;

function draw_stat_graph(canvas_id, tags, bar_data, line_data) {
    const ctx = document.getElementById(canvas_id);
    new Chart(ctx, {
        data: {
        labels: tags,
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

window.onload = function() {
    const middle_data = Array.from(Array(26).keys());
    const btag = {
        "user_id":"kkj9818","tree":10.729559938252086,"data structure":880.0459839506456,"graph":944.770312728214,
        "hash":127.34921184162762,"search":165.79476614862824,"dp":435.3592561963338,"greedy":198.4281883655579,
        "math":340.02889057170825,"bruteforce":212.4184219183956,"recursion":346.85322252739985,"basics":706.3678163790012};
    const user_bdata = []
    for (var i = 0; i < BIG_TAGS.length; i++) {
        user_bdata.push(parseInt(btag[BIG_TAGS[i]]));
    };
    const naver_btag = {"corporation_name": '네이버', "tree": 68.91456993216241, 
    "data structure": 722.4625808393837, "graph": 959.5836232973572, "hash": 94.30426232610468, 
    "search": 122.77646562648036, "dp": 446.5599599053222, "greedy": 158.3769285220893, 
    "math": 336.9676416169149, "bruteforce": 358.5021229819837, "recursion": 149.80522412747382, "basics": 969.2359432237768}
    const naver_bdata = [];
    for (var i = 0; i < BIG_TAGS.length; i++) {
        naver_bdata.push(parseInt(naver_btag[BIG_TAGS[i]]));
    };
    draw_stat_graph("big_stat", BIG_TAGS, user_bdata, naver_bdata);

    const mtag = {"user_id":"kkj9818","bfs":328.58149541439667,"binary search":104.82989942073884,"bruteforcing":212.4184219183956,"deque":31.29731705326738,"dfs":213.50623285902637,"disjoint set":0.0,"divide and conquer":156.03858401307505,"dp":435.3592561963338,"graphs":402.68258445479097,"greedy":198.4281883655579,"hashing":63.67460592081381,"implementation":208.20170965400536,"math":340.02889057170825,"mitm":0.0,"parametric search":60.9648667278894,"priority queue":106.79656211403172,"queue":45.54846375608103,"recursion":190.8146385143248,"segtree":412.4062898321191,"sorting":321.18655731826004,"stack":168.13771799874814,"string":176.9795494067357,"ternary search":0.0,"tree set":63.67460592081381,"trees":10.729559938252086,"two pointer":9.06307108236639};
    const user_mdata = []
    for (var i = 0; i < MIDDLE_TAGS.length; i++) {
        user_mdata.push(parseInt(mtag[MIDDLE_TAGS[i]]));
    };
    const naver_mtag = {"corporation_name": '네이버', "bfs": 273.2854537316701, "binary search": 90.33598499674828, "bruteforcing": 355.7817745190164, "deque": 22.72301077221241, "dfs": 152.4659394880158, "disjoint set": 31.660490836930784, "divide and conquer": 51.80707591843877, "dp": 446.5599599053222, "graphs": 523.4657958788782, "greedy": 158.3769285220893, "hashing": 43.111658558687424, "implementation": 323.03702549950344, "math": 336.9676416169149, "mitm": 0.0, "parametric search": 18.39274827102757, "priority queue": 51.500062832913535, "queue": 23.70404398335198, "recursion": 91.41425379398216, "segtree": 335.1573657432342, "sorting": 450.1620601239804, "stack": 55.6936120614327, "string": 166.8743132063202, "ternary search": 0.0, "tree set": 49.09896608612465, "trees": 68.91456993216241, "two pointer": 16.137423327945108};
    const naver_mdata = [];
    for (var i = 0; i < MIDDLE_TAGS.length; i++) {
        naver_mdata.push(parseInt(naver_mtag[MIDDLE_TAGS[i]]));
    };
    draw_stat_graph("middle_stat", MIDDLE_TAGS, user_mdata, naver_mdata);
};
