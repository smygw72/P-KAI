<template>
  <div class="home">
    <img alt="Vue logo" src="../assets/logo.jpg" id="home_img"/>
    <HomeTitle/>
  </div>
</template>

<script>
import HomeTitle from "@/components/HomeTitle.vue";

export default {
  name: "Home",
  components: {
    HomeTitle
  }
};
</script>

<style scoped>
#home_img{
  width: 20%;
  height: 20%;
}
</style>
