<template>
<!-- ファイル送信エリア -->
    <ul>
        <li>
            ファイル名：<input type="text" v-model="file_name" placeholder="ファイル名">
        </li>
        <li>
            <input type="file">
        </li>
        <li>
            曲の選択：<select v-model="selected_tag">
                <option disabled value="">Please select one</option>
                <option v-for="(tag,key) in tags" :key="key">{{ tag }}</option>
            </select>
        </li>
    </ul>
    <div id="send_button" @click="assessment_request()">送信する</div>
    <br>
    <div id="score_box">
        <div>
            <span id="score">{{ score }}</span>点
        </div>
        <div>
            （<span>{{ ranking.max }}</span>人中<span id="rank">{{ ranking.rank }}</span>位）
        </div>
    </div>
</template>

<script>
import axios from 'axios'
export default {
name: "AssessmentRequest",
data(){
    return {
        file_name: "",
        tags: ["指定なし", "エリーゼのために", "カノン"],
        selected_tag: "",
        score: "",
        ranking: { max: 1000, rank: 1 }
    }
},
methods: {
    assessment_request(){
        const apiUrl = "https://flhrsaddm1.execute-api.ap-northeast-1.amazonaws.com/dev";

        // apiへの送信情報を定義
        const post_data = { 
            user : "Vue-Api-Tester",
            file_name: this.file_name,
            tag : this.selected_tag
        };
        const headers = {
            headers: {
                "Content-Type": "application/json"
            }
        } 
        // POST送信
        const _this = this;
        axios.post(apiUrl,post_data,headers)
        .then((res) => {
            const res_obj = JSON.parse(res.data.body);
            _this.score = res_obj.score;
        })
        .catch((err) => {
            console.log(err);
        })
    }
}
};

</script>

<style scoped>
ul {
    list-style-type: none;
    text-align: left;
    margin-left: 15%;
}
li {
    margin: 3% auto;
}
#send_button{
    width: 20%;
    cursor: pointer;
    color: #42b983;
    margin: 4% auto;
    border: solid green 1px;
}
#send_button:hover{
    background-color: #42b983;
    color: white;
    border-color: green 1px;
}
#score{
    color: #42b983;
}
#rank{
    color: #42b983;    
}
</style>